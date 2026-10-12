"""阶段 A 适配与规则单元掩膜。修改的数据仅在 B 阶段对象中。"""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'stage_a_internal'))
from model import Box, InternalGaussianModel, build_model


def box_model(shape=(8, 1, 16), size=(0.04, 0.005, 0.08), porosity=0.6):
    return build_model([Box((0, 0, 0), size, shape, porosity)])


def subset(model, keep):
    """删除空气单元，保留原有边的子图；不跨被删单元重新连最近邻。"""
    keep = np.asarray(keep, dtype=bool)
    if keep.shape != model.volume.shape or not keep.any():
        raise ValueError('keep must retain at least one cell')
    new_ids = np.full(len(keep), -1, dtype=int)
    new_ids[keep] = np.arange(keep.sum())
    valid_edges = keep[model.edges].all(axis=1)
    node_fields = ['centers', 'covariance', 'cell_lower', 'cell_upper', 'volume',
                   'porosity', 'saturation', 'domain_id']
    values = {name: getattr(model, name)[keep].copy() for name in node_fields}
    values['edges'] = new_ids[model.edges[valid_edges]]
    for name in ['face_area', 'center_distance', 'edge_normal']:
        values[name] = getattr(model, name)[valid_edges].copy()
    return InternalGaussianModel(**values, water_density=model.water_density)


def boundary_face(model, axis=2, side=1):
    """选最外侧平面上的完整单元面。不是通用曲面边界提取。"""
    if axis not in (0, 1, 2) or side not in (-1, 1):
        raise ValueError('invalid face')
    bounds = model.cell_upper if side == 1 else model.cell_lower
    edge = bounds[:, axis].max() if side == 1 else bounds[:, axis].min()
    ids = np.flatnonzero(np.isclose(bounds[:, axis], edge, rtol=0, atol=1e-12))
    widths = model.cell_upper[ids] - model.cell_lower[ids]
    areas = np.prod(widths[:, [d for d in range(3) if d != axis]], axis=1)
    positions = model.centers[ids].copy()
    positions[:, axis] = edge
    return ids, areas, positions, widths[:, axis] / 2
