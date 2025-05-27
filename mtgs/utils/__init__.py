import numpy as np
from scipy.spatial.distance import cdist

def chamfer_distance(gt: np.ndarray, pred: np.ndarray) -> float:
    r"""
    Calculate Chamfer distance.

    Parameters
    ----------
    gt : np.ndarray
        Curve of (G, N) shape,
        where G is the number of data points,
        and N is the number of dimmensions.
    pred : np.ndarray
        Curve of (P, N) shape,
        where P is the number of points,
        and N is the number of dimmensions.

    Returns
    -------
    float
        Chamfer distance

    """
    assert gt.ndim == pred.ndim == 2 and gt.shape[1] == pred.shape[1]
    if (gt[0] == gt[-1]).all():
        gt = gt[:-1]
    dist_mat = cdist(pred, gt)

    dist_pred = dist_mat.min(-1).mean()
    dist_gt = dist_mat.min(0).mean()

    return (dist_pred + dist_gt) / 2
