#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


@torch.jit.script
def quat_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
    assert quat.shape[-1] == 4, f"Expected quaternion shape [..., 4], got {quat.shape}"
    w, x, y, z = torch.unbind(quat, dim=-1)

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    m00 = 1.0 - 2.0 * (yy + zz)
    m01 = 2.0 * (xy - wz)
    m02 = 2.0 * (xz + wy)
    m10 = 2.0 * (xy + wz)
    m11 = 1.0 - 2.0 * (xx + zz)
    m12 = 2.0 * (yz - wx)
    m20 = 2.0 * (xz - wy)
    m21 = 2.0 * (yz + wx)
    m22 = 1.0 - 2.0 * (xx + yy)

    mat = torch.stack([m00, m01, m02, m10, m11, m12, m20, m21, m22], dim=-1)
    return mat.reshape(quat.shape[:-1] + (3, 3))

def random_quat_tensor(N):
    """
    Defines a random quaternion tensor of shape (N, 4)
    """
    u = torch.rand(N)
    v = torch.rand(N)
    w = torch.rand(N)
    return torch.stack(
        [
            torch.sqrt(1 - u) * torch.sin(2 * math.pi * v),
            torch.sqrt(1 - u) * torch.cos(2 * math.pi * v),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
            torch.sqrt(u) * torch.cos(2 * math.pi * w),
        ],
        dim=-1,
    )

@torch.jit.script
def quat_mult(q1, q2):
    # NOTE:
    # Q1 is the quaternion that rotates the vector from the original position to the final position
    # Q2 is the quaternion that been rotated
    w1, x1, y1, z1 = torch.unbind(q1, dim=-1)
    w2, x2, y2, z2 = torch.unbind(q2, dim=-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)

def num_sh_bases(degree: int):
    if degree == 0:
        return 1
    if degree == 1:
        return 4
    if degree == 2:
        return 9
    if degree == 3:
        return 16
    return 25

def RGB2SH(rgb):
    """
    Converts from RGB values [0,1] to the 0th spherical harmonic coefficient
    """
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    """
    Converts from the 0th spherical harmonic coefficient to RGB values [0,1]
    """
    C0 = 0.28209479177387814
    return sh * C0 + 0.5

def qeuler(q, epsilon=0):
    """
    Convert quaternion(s) q to Euler angles.
    Expects a tensor of shape (*, 4), in w, x, y, z order, where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3). in yaw, pitch, roll order.
    """
    assert q.shape[-1] == 4

    original_shape = list(q.shape)
    original_shape[-1] = 3
    q = q.view(-1, 4)

    q0 = q[:, 0]
    q1 = q[:, 1]
    q2 = q[:, 2]
    q3 = q[:, 3]

    yaw = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2*(q2 * q2 + q3 * q3))
    pitch = torch.asin(torch.clamp(2 * (q1 * q3 + q0 * q2), -1+epsilon, 1-epsilon))
    roll = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2*(q1 * q1 + q2 * q2))

    return torch.stack((yaw, pitch, roll), dim=1).view(original_shape)

def rotate_vector_to_vector(v1: Tensor, v2: Tensor):
    """
    Returns a rotation matrix that rotates v1 to align with v2.
    """
    assert v1.dim() == v2.dim()
    assert v1.shape[-1] == 3
    if v1.dim() == 1:
        v1 = v1[None, ...]
        v2 = v2[None, ...]
    N = v1.shape[0]

    u = v1 / torch.norm(v1, dim=-1, keepdim=True)
    Ru = v2 / torch.norm(v2, dim=-1, keepdim=True)
    I = torch.eye(3, 3, device=v1.device).unsqueeze(0).repeat(N, 1, 1)

    # the cos angle between the vectors
    c = torch.bmm(u.view(N, 1, 3), Ru.view(N, 3, 1)).squeeze(-1)

    eps = 1.0e-10
    # the cross product matrix of a vector to rotate around
    K = torch.bmm(Ru.unsqueeze(2), u.unsqueeze(1)) - torch.bmm(
        u.unsqueeze(2), Ru.unsqueeze(1)
    )
    # Rodrigues' formula
    ans = I + K + (K @ K) / (1 + c)[..., None]
    same_direction_mask = torch.abs(c - 1.0) < eps
    same_direction_mask = same_direction_mask.squeeze(-1)
    opposite_direction_mask = torch.abs(c + 1.0) < eps
    opposite_direction_mask = opposite_direction_mask.squeeze(-1)
    ans[same_direction_mask] = torch.eye(3, device=v1.device)
    ans[opposite_direction_mask] = -torch.eye(3, device=v1.device)
    return ans

def matrix_to_quaternion(rotation_matrix: Tensor):
    """
    Convert a 3x3 rotation matrix to a unit quaternion.
    """
    if rotation_matrix.dim() == 2:
        rotation_matrix = rotation_matrix[None, ...]
    assert rotation_matrix.shape[1:] == (3, 3)

    traces = torch.vmap(torch.trace)(rotation_matrix)
    quaternion = torch.zeros(
        rotation_matrix.shape[0],
        4,
        dtype=rotation_matrix.dtype,
        device=rotation_matrix.device,
    )
    for i in range(rotation_matrix.shape[0]):
        matrix = rotation_matrix[i]
        trace = traces[i]
        if trace > 0:
            S = torch.sqrt(trace + 1.0) * 2
            w = 0.25 * S
            x = (matrix[2, 1] - matrix[1, 2]) / S
            y = (matrix[0, 2] - matrix[2, 0]) / S
            z = (matrix[1, 0] - matrix[0, 1]) / S
        elif (matrix[0, 0] > matrix[1, 1]) and (matrix[0, 0] > matrix[2, 2]):
            S = torch.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
            w = (matrix[2, 1] - matrix[1, 2]) / S
            x = 0.25 * S
            y = (matrix[0, 1] + matrix[1, 0]) / S
            z = (matrix[0, 2] + matrix[2, 0]) / S
        elif matrix[1, 1] > matrix[2, 2]:
            S = torch.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
            w = (matrix[0, 2] - matrix[2, 0]) / S
            x = (matrix[0, 1] + matrix[1, 0]) / S
            y = 0.25 * S
            z = (matrix[1, 2] + matrix[2, 1]) / S
        else:
            S = torch.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
            w = (matrix[1, 0] - matrix[0, 1]) / S
            x = (matrix[0, 2] + matrix[2, 0]) / S
            y = (matrix[1, 2] + matrix[2, 1]) / S
            z = 0.25 * S

        quaternion[i] = torch.tensor(
            [w, x, y, z], dtype=matrix.dtype, device=matrix.device
        )
    return quaternion

def interpolate_quats(q1, q2, fraction=0.5):
    """
    Interpolate between two quaternions using spherical linear interpolation (slerp).
    """
    if q1.dim() == 1:
        q1 = q1[None, ...]
    if q2.dim() == 1:
        q2 = q2[None, ...]

    q1 = q1 / torch.norm(q1, dim=-1, keepdim=True)
    q2 = q2 / torch.norm(q2, dim=-1, keepdim=True)

    dot = (q1 * q2).sum(dim=-1, keepdim=True)
    dot = torch.clamp(dot, -1, 1)

    q2 = torch.where(dot < 0, -q2, q2)
    dot = torch.abs(dot)

    similar_mask = dot > 0.9995
    q_interp_similar = q1 + fraction * (q2 - q1)

    theta_0 = torch.acos(dot)
    theta = theta_0 * fraction
    sin_theta = torch.sin(theta)
    sin_theta_0 = torch.sin(theta_0)
    s1 = torch.cos(theta) - dot * sin_theta / sin_theta_0
    s2 = sin_theta / sin_theta_0
    q_interp = (s1[..., None] * q1) + (s2[..., None] * q2)

    final_q_interp = torch.zeros_like(q1)
    final_q_interp = torch.where(similar_mask, q_interp_similar, q_interp)
    final_q_interp = final_q_interp / torch.norm(final_q_interp, dim=-1, keepdim=True)
    return final_q_interp

def get_embedder(multires, i=1):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': i,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class ConditionalDeformNetwork(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, embed_dim=10,
                 x_multires=10, t_multires=10, 
                 deform_quat=True, deform_scale=True):
        super(ConditionalDeformNetwork, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.embed_dim = embed_dim
        self.deform_quat = deform_quat
        self.deform_scale = deform_scale
        self.skips = [D // 2]

        self.embed_time_fn, time_input_ch = get_embedder(t_multires, 1)
        self.embed_fn, xyz_input_ch = get_embedder(x_multires, 3)
        self.input_ch = xyz_input_ch + time_input_ch + embed_dim

        self.linear = nn.ModuleList(
            [nn.Linear(self.input_ch, W, dtype=torch.float32)] + [
                nn.Linear(W, W, dtype=torch.float32) if i not in self.skips else nn.Linear(W + self.input_ch, W, dtype=torch.float32)
                for i in range(D - 1)]
        )

        self.gaussian_warp = nn.Linear(W, 3, dtype=torch.float32)
        if self.deform_quat:
            self.gaussian_rotation = nn.Linear(W, 4, dtype=torch.float32)
        if self.deform_scale:
            self.gaussian_scaling = nn.Linear(W, 3, dtype=torch.float32)

    def forward(self, x, t, condition):
        t_emb = self.embed_time_fn(t)
        x_emb = self.embed_fn(x)
        h = torch.cat([x_emb, t_emb, condition], dim=-1)
        for i, l in enumerate(self.linear):
            h = h.to(self.linear[i].weight.dtype)
            h = self.linear[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x_emb, t_emb, condition, h], -1)

        d_xyz = self.gaussian_warp(h)
        scaling, rotation = None, None
        if self.deform_scale: 
            scaling = self.gaussian_scaling(h)
        if self.deform_quat:
            rotation = self.gaussian_rotation(h)

        return d_xyz, rotation, scaling

def IDFT(time, dim, input_normalized=True):
    """
    Computes the inverse discrete Fourier transform of a given time signal.
    """
    if isinstance(time, float):
        time = torch.tensor(time)
    t = time.view(-1, 1)
    idft = torch.zeros(t.shape[0], dim, dtype=t.dtype, device=t.device)
    indices = torch.arange(dim, dtype=torch.int, device=t.device)
    even_indices = indices[::2]
    odd_indices = indices[1::2]
    if input_normalized:
        idft[:, even_indices] = torch.cos(t * even_indices * 2 * math.pi / dim)
        idft[:, odd_indices] = torch.sin(t * (odd_indices + 1) * 2 * math.pi / dim)
    else:
        idft[:, even_indices] = torch.cos(t * even_indices)
        idft[:, odd_indices] = torch.sin(t * (odd_indices + 1))
    return idft

class MLP(nn.Module):
    """A simple MLP with skip connections."""

    def __init__(
        self,
        in_dims: int,
        out_dims: int,
        num_layers: int = 3,
        hidden_dims: Optional[int] = 256,
        skip_connections: Optional[Tuple[int]] = [0],
    ) -> None:
        super().__init__()
        self.in_dims = in_dims
        self.hidden_dims = hidden_dims
        self.n_output_dims = out_dims
        self.num_layers = num_layers
        self.skip_connections = skip_connections
        layers = []
        if self.num_layers == 1:
            layers.append(nn.Linear(in_dims, out_dims))
        else:
            for i in range(self.num_layers - 1):
                if i == 0:
                    layers.append(nn.Linear(in_dims, hidden_dims))
                elif i in skip_connections:
                    layers.append(nn.Linear(in_dims + hidden_dims, hidden_dims))
                else:
                    layers.append(nn.Linear(hidden_dims, hidden_dims))
            layers.append(nn.Linear(hidden_dims, out_dims))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor) -> Tensor:
        input = x
        for i, layer in enumerate(self.layers):
            if i in self.skip_connections:
                x = torch.cat([x, input], -1)
            x = layer(x)
            if i < len(self.layers) - 1:
                x = nn.functional.relu(x)
        return x
