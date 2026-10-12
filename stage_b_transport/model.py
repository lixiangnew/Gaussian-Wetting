"""阶段 A：固定盒体中的内部高斯、精确代表体积和材料邻接图。

仅依赖 NumPy。长度 m、体积 m³、水质量 kg。这里没有水传播求解器。
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class Box:
    """一个均匀材料的已知盒体；不同 Box 默认是独立材料域。

    lower/upper: 世界坐标边界，单位 m。
    shape: 三个方向的划分数，例如 (4, 4, 4) 生成 64 个高斯。
    porosity: 孔隙率，不是渲染 opacity。
    saturation: 初始化含水状态；阶段 A 不更新它。
    """
    lower: tuple
    upper: tuple
    shape: tuple
    porosity: float = 0.6
    saturation: float = 0.0


@dataclass
class InternalGaussianModel:
    centers: np.ndarray             # [N,3] 高斯中心，m
    covariance: np.ndarray          # [N,3,3] 空间协方差，m²
    cell_lower: np.ndarray          # [N,3] 代表体积的下边界，m
    cell_upper: np.ndarray          # [N,3] 代表体积的上边界，m
    volume: np.ndarray              # [N] 不重叠的材料体积，m³
    porosity: np.ndarray            # [N] 孔隙率
    saturation: np.ndarray          # [N] 初始化饱和度
    domain_id: np.ndarray           # [N] 所属独立材料域
    edges: np.ndarray              # [E,2] 无向边，每对只存一次
    face_area: np.ndarray          # [E] 共享面的面积，m²
    center_distance: np.ndarray    # [E] 相邻中心距离，m
    edge_normal: np.ndarray        # [E,3] 从 edges[:,0] 指向 edges[:,1]
    water_density: float = 1000.0   # kg/m³，指定常数

    @property
    def pore_volume(self):
        """每个代表体积中的孔隙空间，而不是高斯椭球体积。"""
        return self.volume * self.porosity

    @property
    def capacity_mass(self):
        """每个节点完全饱和时能容纳的水质量。"""
        return self.water_density * self.pore_volume

    @property
    def water_mass(self):
        """初始水质量；B 阶段应以质量为更新变量，避免两份状态不同步。"""
        return self.capacity_mass * self.saturation

    def component_labels(self):
        """用深度优先遍历给图的连通分量编号；孤立节点也算一个分量。"""
        adjacency = [[] for _ in range(len(self.centers))]
        for i, j in self.edges:
            adjacency[i].append(j)
            adjacency[j].append(i)
        labels = np.full(len(self.centers), -1, dtype=np.int64)
        count = 0
        for start in range(len(labels)):
            if labels[start] != -1:
                continue
            labels[start] = count
            stack = [start]
            while stack:
                i = stack.pop()
                for j in adjacency[i]:
                    if labels[j] == -1:
                        labels[j] = count
                        stack.append(j)
            count += 1
        return labels

    def summary(self):
        """返回可写入 JSON 的总体积、容量和图规模。"""
        labels = self.component_labels()
        return {
            'gaussians': len(self.centers),
            'edges': len(self.edges),
            'components': int(labels.max()) + 1,
            'material_volume_m3': float(self.volume.sum()),
            'pore_volume_m3': float(self.pore_volume.sum()),
            'capacity_kg': float(self.capacity_mass.sum()),
            'initial_water_kg': float(self.water_mass.sum()),
        }

    def save(self, path):
        """保存命名数组；用 np.load(path, allow_pickle=False) 读取。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, centers=self.centers, covariance=self.covariance,
            cell_lower=self.cell_lower, cell_upper=self.cell_upper,
            volume=self.volume, porosity=self.porosity,
            saturation=self.saturation, domain_id=self.domain_id,
            edges=self.edges, face_area=self.face_area,
            center_distance=self.center_distance, edge_normal=self.edge_normal,
            water_density=np.array(self.water_density),
            schema_version=np.array(1),
        )


def _validate_box(box):
    """拒绝无效几何和材料参数，避免静默截断用户输入。"""
    lower = np.asarray(box.lower, dtype=np.float64)
    upper = np.asarray(box.upper, dtype=np.float64)
    shape = np.asarray(box.shape)
    if lower.shape != (3,) or upper.shape != (3,):
        raise ValueError('lower 和 upper 必须各有三个坐标')
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError('边界坐标必须有限')
    if np.any(upper <= lower):
        raise ValueError('每个方向的 upper 都必须大于 lower')
    if shape.shape != (3,) or not np.issubdtype(shape.dtype, np.integer):
        raise ValueError('shape 必须包含三个整数')
    if np.any(shape <= 0):
        raise ValueError('shape 必须为正整数')
    for name, value in [('porosity', box.porosity), ('saturation', box.saturation)]:
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(name + ' 必须在 [0,1] 内')
    if box.porosity == 0 and box.saturation != 0:
        raise ValueError('无孔隙材料的 saturation 应定义为 0')
    return lower, upper, shape.astype(np.int64)


def build_model(boxes, sigma_ratio=0.5, water_density=1000.0):
    """按互不重叠的盒体域构建内部表示。

    sigma_ratio 定义高斯标准差 / 单元边长，只影响协方差。
    同一 Box 内只连接共享面的六邻居，不连接对角邻居。
    不同 Box 之间不自动连边，即使接触；接触界面需后续显式处理。
    """
    boxes = list(boxes)
    if not boxes:
        raise ValueError('至少需要一个 Box')
    if not np.isfinite(sigma_ratio) or sigma_ratio <= 0:
        raise ValueError('sigma_ratio 必须为有限正数')
    if not np.isfinite(water_density) or water_density <= 0:
        raise ValueError('water_density 必须为有限正数')
    validated = [_validate_box(box) for box in boxes]
    # 两个盒体如果在三个方向都有正长度交叠，就会重复计量体积。
    for a in range(len(boxes)):
        for b in range(a):
            overlap = np.minimum(validated[a][1], validated[b][1]) - np.maximum(
                validated[a][0], validated[b][0])
            if np.all(overlap > 0):
                raise ValueError('Box 体积不能重叠；应先划分不重叠的材料域')

    names = ['centers', 'covariance', 'cell_lower', 'cell_upper', 'volume',
             'porosity', 'saturation', 'domain_id', 'edges', 'face_area',
             'center_distance', 'edge_normal']
    chunks = {name: [] for name in names}
    offset = 0
    for domain, (box, (lower, upper, shape)) in enumerate(zip(boxes, validated)):
        # 用共享的边界数组划分体积，邻居使用同一个面坐标。
        axes = [np.linspace(lower[d], upper[d], shape[d] + 1) for d in range(3)]
        ijk = np.indices(tuple(shape)).reshape(3, -1).T
        cell_lower = np.column_stack([axes[d][ijk[:, d]] for d in range(3)])
        cell_upper = np.column_stack([axes[d][ijk[:, d] + 1] for d in range(3)])
        widths = cell_upper - cell_lower
        centers = (cell_lower + cell_upper) / 2
        count = len(centers)
        covariance = np.zeros((count, 3, 3), dtype=np.float64)
        for d in range(3):
            covariance[:, d, d] = (sigma_ratio * widths[:, d]) ** 2
        chunks['centers'].append(centers)
        chunks['covariance'].append(covariance)
        chunks['cell_lower'].append(cell_lower)
        chunks['cell_upper'].append(cell_upper)
        chunks['volume'].append(np.prod(widths, axis=1))
        chunks['porosity'].append(np.full(count, box.porosity, dtype=np.float64))
        chunks['saturation'].append(np.full(count, box.saturation, dtype=np.float64))
        chunks['domain_id'].append(np.full(count, domain, dtype=np.int64))

        ids = np.arange(count).reshape(tuple(shape))
        # 只处理 +x、+y、+z，防止把同一条无向边保存两次。
        for d in range(3):
            left, right = [slice(None)] * 3, [slice(None)] * 3
            left[d], right[d] = slice(0, -1), slice(1, None)
            i, j = ids[tuple(left)].ravel(), ids[tuple(right)].ravel()
            edges = np.column_stack((i, j)) + offset
            normal = np.zeros((len(i), 3))
            normal[:, d] = 1.0
            other_axes = [axis for axis in range(3) if axis != d]
            chunks['edges'].append(edges)
            chunks['face_area'].append(np.prod(widths[i][:, other_axes], axis=1))
            chunks['center_distance'].append(centers[j, d] - centers[i, d])
            chunks['edge_normal'].append(normal)
        offset += count

    arrays = {name: np.concatenate(parts, axis=0) for name, parts in chunks.items()}
    return InternalGaussianModel(**arrays, water_density=float(water_density))
