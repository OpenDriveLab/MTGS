"""
Compatibility shim that adapts gsplat-mps (v0.1.3) to the gsplat v1.4.0 API
used by MTGS. Import this module before any MTGS code that uses gsplat.
"""
import math
import os
import types

import torch

import gsplat

_DEBUG = os.environ.get("GSPLAT_MPS_DEBUG") == "1"


def _debug_print(*args, **kwargs):
    if _DEBUG:
        print(*args, **kwargs)


# --- Shim: gsplat.cuda._wrapper.spherical_harmonics ---
# MTGS imports: from gsplat.cuda._wrapper import spherical_harmonics
# gsplat-mps exposes: gsplat.spherical_harmonics (same signature)

_cuda_mod = types.ModuleType("gsplat.cuda")
_wrapper_mod = types.ModuleType("gsplat.cuda._wrapper")
_wrapper_mod.spherical_harmonics = gsplat.spherical_harmonics
_cuda_mod._wrapper = _wrapper_mod

gsplat.cuda = _cuda_mod
import sys
sys.modules["gsplat.cuda"] = _cuda_mod
sys.modules["gsplat.cuda._wrapper"] = _wrapper_mod


# --- Shim: gsplat.rendering.rasterization ---
# Adapts the gsplat v1.4.0 high-level rasterization() call to the
# gsplat-mps v0.1.3 low-level project_gaussians + rasterize_gaussians API.

def _rasterization_mps(
    means,
    quats,
    scales,
    opacities,
    colors,
    viewmats,
    Ks,
    width,
    height,
    tile_size=16,
    packed=False,
    near_plane=0.01,
    far_plane=1e10,
    render_mode="RGB",
    sparse_grad=False,
    absgrad=False,
    rasterize_mode="classic",
    **kwargs,
):
    """Adapter: gsplat v1.4.0 rasterization() -> gsplat-mps v0.1.3 low-level API."""
    import sys
    device = means.device

    # Squeeze batch dim if present (B=1). MTGS may pass (N,3) or (1,N,3).
    if means.ndim == 3:
        means = means[0]
    if quats.ndim == 3:
        quats = quats[0]
    if scales.ndim == 3:
        scales = scales[0]
    if opacities.ndim == 2 and opacities.shape[0] == 1:
        opacities = opacities[0]
    if opacities.ndim == 1:
        opacities = opacities.unsqueeze(-1)  # (N,) -> (N, 1)
    if colors.ndim == 3:
        colors = colors[0]
    viewmat = viewmats[0]  # (4, 4)
    K = Ks[0]              # (3, 3)

    N = means.shape[0]
    _debug_print(f"[MPS rasterize] N={N}, {width}x{height}, mode={render_mode}", flush=True, file=sys.stderr)

    # Extract intrinsic scalars
    fx = K[0, 0].item()
    fy = K[1, 1].item()
    cx = K[0, 2].item()
    cy = K[1, 2].item()

    # Tile bounds
    BLOCK = tile_size
    tile_bounds = (
        (width + BLOCK - 1) // BLOCK,
        (height + BLOCK - 1) // BLOCK,
        1,
    )

    # Step 1: Project with the gsplat v1.4.0 torch-equivalent formulas.
    #
    # The older gsplat-mps projection path is not equivalent to v1.4's
    # fully_fused_projection() for MTGS' camera model, and its upstream test has
    # the reference comparison commented out as "TODO: failing". Keep Metal for
    # the 2D rasterizer, but compute radii/means2d/depths/conics here to match
    # v1.4 semantics more closely.
    _debug_print(f"[MPS] Step 1: v1.4 torch projection...", flush=True, file=sys.stderr)
    eps2d = float(kwargs.get("eps2d", 0.3))
    radius_clip = float(kwargs.get("radius_clip", 0.0) or 0.0)

    quat_norm = quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    qw, qx, qy, qz = quat_norm.unbind(dim=-1)
    rot = torch.stack(
        [
            torch.stack([1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)], dim=-1),
            torch.stack([2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)], dim=-1),
            torch.stack([2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)], dim=-1),
        ],
        dim=-2,
    )
    covars = (rot * scales.unsqueeze(-2)) @ (rot * scales.unsqueeze(-2)).transpose(-1, -2)

    R = viewmat[:3, :3]
    t = viewmat[:3, 3]
    means_c = means @ R.T + t
    covars_c = torch.einsum("ij,njk,lk->nil", R, covars, R)
    depths = means_c[:, 2]

    tx, ty, tz = means_c.unbind(dim=-1)
    tz_safe = tz.clamp_min(1e-10)
    tan_fovx = 0.5 * width / fx
    tan_fovy = 0.5 * height / fy
    lim_x_pos = (width - cx) / fx + 0.3 * tan_fovx
    lim_x_neg = cx / fx + 0.3 * tan_fovx
    lim_y_pos = (height - cy) / fy + 0.3 * tan_fovy
    lim_y_neg = cy / fy + 0.3 * tan_fovy
    tx_clamped = tz_safe * torch.clamp(tx / tz_safe, min=-lim_x_neg, max=lim_x_pos)
    ty_clamped = tz_safe * torch.clamp(ty / tz_safe, min=-lim_y_neg, max=lim_y_pos)

    zeros = torch.zeros_like(tz_safe)
    J = torch.stack(
        [
            torch.stack([torch.full_like(tz_safe, fx) / tz_safe, zeros, -fx * tx_clamped / (tz_safe * tz_safe)], dim=-1),
            torch.stack([zeros, torch.full_like(tz_safe, fy) / tz_safe, -fy * ty_clamped / (tz_safe * tz_safe)], dim=-1),
        ],
        dim=-2,
    )
    cov2d_orig = J @ covars_c @ J.transpose(-1, -2)
    det_orig = cov2d_orig[:, 0, 0] * cov2d_orig[:, 1, 1] - cov2d_orig[:, 0, 1] * cov2d_orig[:, 1, 0]
    cov2d = cov2d_orig + torch.eye(2, device=device, dtype=means.dtype) * eps2d
    det = (cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] * cov2d[:, 1, 0]).clamp_min(1e-10)
    conics = torch.stack(
        [
            cov2d[:, 1, 1] / det,
            -(cov2d[:, 0, 1] + cov2d[:, 1, 0]) * 0.5 / det,
            cov2d[:, 0, 0] / det,
        ],
        dim=-1,
    )
    b = 0.5 * (cov2d[:, 0, 0] + cov2d[:, 1, 1])
    radius = torch.ceil(3.0 * torch.sqrt(b + torch.sqrt(torch.clamp(b * b - det, min=0.01))))
    xys = torch.stack([fx * tx / tz_safe + cx, fy * ty / tz_safe + cy], dim=-1)

    valid = (det_orig > 0) & (depths > near_plane) & (depths < far_plane)
    valid = valid & (radius > radius_clip)
    inside = (
        (xys[:, 0] + radius > 0)
        & (xys[:, 0] - radius < width)
        & (xys[:, 1] + radius > 0)
        & (xys[:, 1] - radius < height)
    )
    radius = torch.where(valid & inside, radius, torch.zeros_like(radius))
    radii = radius.to(torch.int32)

    tile_center = xys / float(BLOCK)
    tile_radius = radius.unsqueeze(-1) / float(BLOCK)
    tile_min = torch.trunc(tile_center - tile_radius).to(torch.int32).clamp_min(0)
    tile_max = torch.trunc(tile_center + tile_radius + 1.0).to(torch.int32).clamp_min(0)
    tile_min[:, 0].clamp_(max=tile_bounds[0])
    tile_min[:, 1].clamp_(max=tile_bounds[1])
    tile_max[:, 0].clamp_(max=tile_bounds[0])
    tile_max[:, 1].clamp_(max=tile_bounds[1])
    num_tiles_hit = ((tile_max[:, 0] - tile_min[:, 0]) * (tile_max[:, 1] - tile_min[:, 1])).to(torch.int32)
    num_tiles_hit = torch.where(radii > 0, num_tiles_hit, torch.zeros_like(num_tiles_hit))

    compensations = torch.sqrt(torch.clamp(det_orig / det, min=0.0)) if rasterize_mode == "antialiased" else None

    _debug_print(f"[MPS] project done: xys={xys.shape}, depths={depths.shape}", flush=True, file=sys.stderr)
    _debug_print(f"[MPS] xys range: [{xys.min().item():.1f}, {xys.max().item():.1f}]", flush=True, file=sys.stderr)
    _debug_print(f"[MPS] depths range: [{depths.min().item():.3f}, {depths.max().item():.3f}]", flush=True, file=sys.stderr)
    _debug_print(f"[MPS] radii range: [{radii.min().item():.3f}, {radii.max().item():.3f}]", flush=True, file=sys.stderr)
    _debug_print(f"[MPS] num_tiles_hit range: [{num_tiles_hit.min().item()}, {num_tiles_hit.max().item()}]", flush=True, file=sys.stderr)
    _debug_print(f"[MPS] colors range: [{colors.min().item():.3f}, {colors.max().item():.3f}]", flush=True, file=sys.stderr)
    _debug_print(f"[MPS] opacities range: [{opacities.min().item():.3f}, {opacities.max().item():.3f}]", flush=True, file=sys.stderr)

    raster_opacities = opacities
    if compensations is not None:
        raster_opacities = opacities * compensations.unsqueeze(-1)
        _debug_print(
            f"[MPS] antialias compensation range: [{compensations.min().item():.3f}, {compensations.max().item():.3f}], "
            f"mean={compensations.mean().item():.3f}",
            flush=True,
            file=sys.stderr,
        )

    # Step 2: Prepare colors (append depth if needed)
    # Clamp RGB channels to [0, 1] — gsplat v1.4.0 rasterization() handles this
    # internally, but rasterize_gaussians expects pre-normalized colors.
    rgb_channels = colors[:, :3].clamp(0.0, 1.0)
    extra_channels = colors[:, 3:]  # normals, etc. — keep as-is
    render_colors = torch.cat([rgb_channels, extra_channels], dim=-1) if extra_channels.shape[-1] > 0 else rgb_channels
    if render_mode == "RGB+ED":
        render_colors = torch.cat([render_colors, depths.unsqueeze(-1)], dim=-1)

    # Step 3: Rasterize
    #
    # gsplat v1.4 rasterization() returns premultiplied color/depth plus a
    # separate alpha; MTGS composites the background after this call. The
    # low-level gsplat-mps rasterizer defaults background=None to white, so pass
    # an explicit zero background to avoid baking a white background into the
    # render and to keep alpha meaningful.
    _debug_print(f"[MPS] Step 3: rasterize_gaussians...", flush=True, file=sys.stderr)
    zero_background = torch.zeros(
        render_colors.shape[-1], device=device, dtype=render_colors.dtype
    )
    render_img, alpha_img = gsplat.rasterize_gaussians(
        xys=xys,
        depths=depths,
        radii=radii,
        conics=conics,
        num_tiles_hit=num_tiles_hit,
        colors=render_colors,
        opacity=raster_opacities,
        img_height=height,
        img_width=width,
        background=zero_background,
        return_alpha=True,
    )  # (H, W, C), (H, W)
    if num_tiles_hit.sum().item() == 0:
        alpha_img = torch.zeros_like(alpha_img)
    alpha = alpha_img.unsqueeze(0).unsqueeze(-1)  # (1, H, W, 1)

    # gsplat's "ED" modes return expected depth: sum(w_i z_i) / sum(w_i).
    # Rendering depth as an ordinary feature gives sum(w_i z_i), so normalize
    # the appended depth channel by alpha before returning it.
    if render_mode == "RGB+ED":
        render_img[..., -1:] = torch.where(
            alpha_img.unsqueeze(-1) > 0,
            render_img[..., -1:] / alpha_img.clamp_min(1e-8).unsqueeze(-1),
            render_img[..., -1:],
        )

    _debug_print(f"[MPS] rasterize done: {render_img.shape}", flush=True, file=sys.stderr)
    _debug_print(f"[MPS] render_img RGB range: [{render_img[..., :3].min().item():.3f}, {render_img[..., :3].max().item():.3f}]", flush=True, file=sys.stderr)
    _debug_print(f"[MPS] render_img all range: [{render_img.min().item():.3f}, {render_img.max().item():.3f}]", flush=True, file=sys.stderr)
    _debug_print(f"[MPS] render_img unique RGB values (sample): {render_img[0, 0, :3].tolist()}", flush=True, file=sys.stderr)

    # Add batch dim back: (H, W, C) -> (1, H, W, C)
    render_img = render_img.unsqueeze(0)

    # Build info dict matching gsplat v1.4.0 output
    info = {
        "radii": radii.unsqueeze(0),  # (1, N) — MTGS expects this shape
        "means2d": xys.unsqueeze(0),  # (1, N, 2)
    }

    return render_img, alpha, info


_rendering_mod = types.ModuleType("gsplat.rendering")
_rendering_mod.rasterization = _rasterization_mps

gsplat.rendering = _rendering_mod
sys.modules["gsplat.rendering"] = _rendering_mod


# --- Shim: gsplat.strategy (stub) ---
# nerfstudio's splatfacto imports gsplat.strategy.DefaultStrategy.
# We only need it to be importable; it's not used during rendering.

class _DefaultStrategyStub:
    """Stub for gsplat.strategy.DefaultStrategy — not available in gsplat-mps v0.1.3."""
    def __init__(self, *args, **kwargs):
        pass
    def __getattr__(self, name):
        raise NotImplementedError("gsplat.strategy is a stub in MPS mode; training is not supported.")

_strategy_mod = types.ModuleType("gsplat.strategy")
_strategy_mod.DefaultStrategy = _DefaultStrategyStub

gsplat.strategy = _strategy_mod
sys.modules["gsplat.strategy"] = _strategy_mod


_debug_print("[gsplat_mps_compat] gsplat-mps compatibility shim loaded")
