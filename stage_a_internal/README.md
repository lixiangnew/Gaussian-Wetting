# 阶段 A：固定几何中的内部高斯表示

对应计划书第 6 节：**在已知固定几何中构建内部高斯、代表体积和连通图；改变高斯数量后，总体积与孔隙容量基本不变。**

这里实现第一版规则盒体基线：用不重叠的长方体单元分配物理体积，并在每个单元中心放置空间高斯。它是“具有明确代表体积的内部高斯表示”，底层代表域仍是规则划分，不宣称已经实现任意非规则高斯的守恒离散或论文创新。

## 文件

| 文件 | 用途 |
|---|---|
| [model.py](model.py) | 核心数据结构、构建算法、连通分量和保存 |
| [run_demo.py](run_demo.py) | 三组验收实验与剖面示意图 |
| [test_model.py](test_model.py) | 7 项测试：几何、体积、容量、连通性和保存 |
| [CODE_GUIDE.md](CODE_GUIDE.md) | 按执行顺序解释代码与 NumPy 操作 |
| outputs/ | NPZ 数据、CSV 对照、JSON 报告、SVG 示意图 |

仅依赖 NumPy，不需要 CUDA、PyTorch、Warp、训练照片或 GaussianFluent。使用本机已有环境，不修改上游代码。

## 运行

在项目根目录 PowerShell 执行：

```powershell
python stage_a_internal/run_demo.py
python -m unittest discover -s stage_a_internal -p test_model.py -v
```

其他有 NumPy 的环境也可以用 `python` 替换上面的解释器。默认输出固定在本文件夹的 outputs 中，重复运行会重写同名演示输出。

## 实测结果（2026-10-08，CPU）

固定几何：10 × 6 × 4 cm。孔隙率 φ=0.6，初始饱和度 S=0，指定水密度 1000 kg/m³。

| 划分 | 内部高斯数 | 图边数 | 材料总体积 | 总孔隙体积 | 满容量水质量 |
|---|---:|---:|---:|---:|---:|
| 4 × 4 × 4 | 64 | 144 | 240 cm³ | 144 cm³ | 144 g |
| 8 × 8 × 8 | 512 | 1,344 | 240 cm³ | 144 cm³ | 144 g |
| 16 × 16 × 16 | 4,096 | 11,520 | 240 cm³ | 144 cm³ | 144 g |

体积和容量相对误差均小于 1e-12。7 项单元测试全部通过。这里的 144 g 是设定几何与孔隙率得到的理论最大容量，不是已经吸入的水量或实测材料性能；当前初始水量为 0。

附加验收：

- sigma_ratio 从 0.3 改到 1.2：协方差扩大 16 倍，容量和图边不变。
- 中间留出 1 cm 空气间隙：材料总体积变成 216 cm³、容量 129.6 g，两个连通分量，跨域边为 0。此例改变了实际几何，不能与无间隙块体要求相同容量。
- 零孔隙率材料容量为 0；一个节点时图边为空；重叠盒体被拒绝，防止体积重复计量。

详细结果：[report.json](outputs/report.json)；[分辨率对照 CSV](outputs/resolution_check.csv)。

图示：[64 个高斯的第一层](outputs/block_4_slice.svg)、[512 个高斯的第一层](outputs/block_8_slice.svg)、[带空气间隙的两块材料](outputs/air_gap_slice.svg)。灰色方框是代表体积的剖面，蓝色椭圆是高斯 1σ 轮廓，橙色线是图边；不是 GS 渲染结果，也不是水传播结果。

## 如何修改自己的例子

在本目录下的脚本中使用：

```python
from model import Box, build_model

box = Box(
    lower=(0, 0, 0),
    upper=(0.10, 0.06, 0.04),  # 米
    shape=(8, 8, 8),           # 每个方向分成几块
    porosity=0.6,
    saturation=0.0,
)
model = build_model([box], sigma_ratio=0.5)
print(model.summary())
model.save('my_material.npz')
```

shape 控制高斯数；porosity 控制容量；sigma_ratio 只控制高斯宽度。固定几何、孔隙率，改变 shape 或 sigma_ratio 不应改变总体容量。

NPZ 读取方法：

```python
import numpy as np

with np.load('outputs/block_8.npz', allow_pickle=False) as data:
    centers = data['centers']     # [512,3]，单位 m
    volume = data['volume']       # [512]，单位 m³
    porosity = data['porosity']   # [512]
    edges = data['edges']         # [1344,2]，节点索引
    capacity = float(data['water_density']) * volume * porosity
```

路径相对运行时的当前目录；上例假定当前目录为 stage_a_internal。

## 当前范围与 B 阶段接口

支持已知、轴对齐、互不重叠的盒体材料域，域内孔隙率均匀。不同 Box 默认不连边，即使接触，因此不能通过堆叠多个 Box 自动模拟连通复合材料。该策略显式阻止跨独立物体传水，后续需为可信的接触材料界面增加规则。

高斯核在数学上无限延伸；cell_lower/cell_upper 定义有限的物理代表域。不能把核尾巴伸出边界理解成边界外还有材料。本阶段未实现连续场插值，未来查询时必须尊重材料域和连通性。

没有读取网格模型或重建 PLY，没有非规则采样、裂缝自动识别、渗透率/毛细模型、传播、动态重采样、表面储水、湿润光学或 CUDA 渲染。NPZ 不是官方 3DGS 可直接渲染的 PLY，它没有外观系数与 opacity；阶段 A 不需要编造这些属性。

B 阶段可复用 centers、volume、porosity、edges、face_area、center_distance、edge_normal。它需要新增节点水质量更新、材料渗透率、毛细关系与边界账本。当前边只是几何连接，不是已经计算好的水力导通；尤其零孔隙率或不透水材料，必须由 B 阶段明确禁止通量，不能仅凭有边就允许流动。

分辨率验收只证明总体积和容量正确，尚不能证明不同分辨率下的传播速度一致或输运解收敛。
