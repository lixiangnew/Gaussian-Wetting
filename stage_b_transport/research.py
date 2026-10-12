"""候选研究方向：交叠体积守恒转移、体积加权且有域约束的高斯查询。

这些是研究原型与一致性基线，不宣称相关数学思想本身具有新颖性。
"""
import numpy as np


def conservative_remap(old, new, old_mass):
    """对同一材料域的新旧正交代表单元做精确交叠积分。

    old_mass 在旧单元内按均匀体积密度解释。不根据高斯核复制水量。
    必须覆盖相同几何与相同 domain_id 标签；不支持变形/移动材料域。
    """
    old_mass = np.asarray(old_mass, dtype=float)
    if old_mass.shape != old.volume.shape or not np.isfinite(old_mass).all():
        raise ValueError('invalid old mass')
    if (old_mass < 0).any() or (old_mass > old.capacity_mass + 1e-15).any():
        raise ValueError('old mass outside capacity')
    new_pore = np.zeros_like(new.volume)
    new_mass = np.zeros_like(new.volume)
    covered = np.zeros_like(old.volume)
    for j in range(len(new.volume)):
        widths = np.maximum(np.minimum(old.cell_upper, new.cell_upper[j]) -
                            np.maximum(old.cell_lower, new.cell_lower[j]), 0)
        overlap = np.prod(widths, axis=1)
        overlap[old.domain_id != new.domain_id[j]] = 0
        if not np.isclose(overlap.sum(), new.volume[j], rtol=1e-10, atol=1e-18):
            raise ValueError('new cell not covered by matching old material domain')
        covered += overlap
        new_pore[j] = overlap @ old.porosity
        new_mass[j] = overlap @ (old_mass / old.volume)
    if not np.allclose(covered, old.volume, rtol=1e-10, atol=1e-18):
        raise ValueError('old domain not fully covered, or new cells overlap')
    new_phi = new_pore / new.volume
    new_capacity = new.water_density*new_pore
    if old.water_density != new.water_density:
        raise ValueError('density changes require a separate physical model')
    new_s = np.divide(new_mass, new_capacity, out=np.zeros_like(new_mass), where=new_capacity > 0)
    return new_phi, new_mass, new_s


def query(centers, covariance, volume, porosity, saturation, labels, points,
          point_labels, volume_weighted=True, direct_s=False):
    """按给定连通域查询 φ、θ=φS、S。域归属应由几何获得，不由核尾猜测。

    权重为 V_i exp(−Mahalanobis²/2)，逐点归一化。不是容量积分或渲染器。
    只查询已知属于域内部的点；标签无候选节点时报错。
    """
    inv = np.linalg.inv(covariance)
    result = []
    for point, label in zip(points, point_labels):
        chosen = np.flatnonzero(labels == label)
        if not len(chosen):
            raise ValueError('query has no material support')
        delta = centers[chosen]-point
        logw = -0.5*np.einsum('ni,nij,nj->n', delta, inv[chosen], delta)
        if volume_weighted:
            logw += np.log(volume[chosen])
        weight = np.exp(logw-logw.max())
        weight /= weight.sum()
        phi = float(weight @ porosity[chosen])
        theta = float(weight @ (porosity[chosen]*saturation[chosen]))
        s = float(weight @ saturation[chosen]) if direct_s else (theta/phi if phi > 0 else 0.0)
        result.append((phi, theta, s))
    return np.asarray(result)
