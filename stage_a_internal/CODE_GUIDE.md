# 一步一步读阶段 A 的代码

先只看三个对象：**Box 是输入几何；InternalGaussianModel 是输出数据；build_model 是把输入转换成输出的函数。** 不需要先掌握 3DGS 或 MPM 源码。

## 1. 这次到底在构造什么

设一个盒体长宽高分别为 Lx、Ly、Lz，把三个方向分成 nx、ny、nz 份。内部节点数为：

```text
N = nx × ny × nz
dx = Lx / nx
dy = Ly / ny
dz = Lz / nz
```

每个小块拥有一个高斯，但小块与高斯不是同一件事：小块是有限、互不重叠的物理代表域；高斯是有中心、协方差的平滑空间核。

数学定义可以写为 g_i(x)=exp(−0.5 (x−μ_i)^T Σ_i^−1 (x−μ_i))。本阶段保存 μ_i、Σ_i，不必在每个位置实际计算 g_i，因为容量与图构造不依赖采样核值。

高斯中心放在小块中心，标准差设为边长的 sigma_ratio 倍。默认 sigma_ratio=0.5，1σ 椭球接触小块的各个面；它的角部并不填满小块，核在 1σ 外也不是零。

代表体积直接定义为 V_i=dx·dy·dz，不能换成 4πσxσyσz/3，也不能把高斯积分当作材料体积。

## 2. Python 从哪里开始执行

运行 `python run_demo.py` 时：

1. 导入标准库和 NumPy。
2. `from model import Box, build_model` 从同目录 model.py 导入这两个名字。
3. 登记 write_slice_svg 和 main 的函数定义，此时没有执行函数体。
4. 到文件末尾 `if __name__ == '__main__': main()`，执行 main。
5. main 构造 Box，调用 build_model，再检查结果并写文件。

`if __name__ ...` 的作用是：直接运行脚本时启动演示，被其他文件导入时不自动运行演示。

## 3. 输入 Box：为什么这样写

```python
@dataclass(frozen=True)
class Box:
    lower: tuple
    upper: tuple
    shape: tuple
    porosity: float = 0.6
    saturation: float = 0.0
```

`dataclass` 来自 Python 标准库。它自动生成初始化方法，让我们可以写 `Box(lower=..., upper=..., shape=...)`，而不必手写逐项赋值。`frozen=True` 防止创建后重新给这些字段赋值。

`tuple` 表示元组，例如 `(0,0,0)`；`float` 表示浮点数。这些类型提示帮助阅读，但不会自动保证输入正确，因此另有验证函数。

例如 `Box((0,0,0), (0.10,0.06,0.04), (4,4,4))` 表示一个以米为单位的盒体，不是从照片中恢复的物体。

## 4. _validate_box：先检查，再计算

`np.asarray(..., dtype=np.float64)` 把坐标变成双精度数组。双精度主要用于降低几何与体积累计中的舍入误差，不代表物理参数就有这么高的实验精度。

`lower.shape != (3,)` 检查是否恰好三个数。这里数组的 shape 是形状，而 Box.shape 是三个方向的划分数，两者名字相同但含义不同。

`np.isfinite` 排除 NaN 和无穷大；`np.all` 表示全部满足；`np.any` 表示至少一个满足。`np.any(upper <= lower)` 检查是否存在零长度或负长度的方向。

`np.issubdtype(shape.dtype, np.integer)` 确认划分数为整数，防止把 2.5 静默截断成 2。随后检查它们都大于零。

孔隙率和饱和度必须在 [0,1]。φ=0 时这里要求 S=0，符合计划书对无孔隙节点的约定。`raise ValueError(...)` 意味着停止当前构建并报告明确原因，不悄悄修正输入。

函数返回校验后的 lower、upper、shape，供 build_model 使用。

## 5. 为什么还要检查盒体之间是否重叠

对于盒体 a、b，每个方向的交叠长度为：

```python
overlap = np.minimum(upper_a, upper_b) - np.maximum(lower_a, lower_b)
```

三个方向的交叠长度全为正，才存在正体积交集。构建器拒绝这种输入，否则同一块材料会被算两遍，容量就错了。共享一个面、边或顶点的交叠体积为零，不触发此错误，但也不会自动连边。

## 6. np.linspace：先精确列出切分边界

以 x∈[0,0.10]、nx=4 为例：

```python
np.linspace(0, 0.10, 5)
# [0.000, 0.025, 0.050, 0.075, 0.100]
```

`linspace(start, stop, count)` 在两端之间生成 count 个等距数。4 个区间需要 5 个边界，因此代码使用 `shape[d] + 1`。

三个方向分别得到边界数组，放到列表 axes 里。相邻单元从同一个数组取共享面坐标，避免分别计算左右边界导致不一致。

## 7. np.indices、reshape、T：给每块材料编号

```python
ijk = np.indices(tuple(shape)).reshape(3, -1).T
```

假设 shape=(2,2,2)：

- `np.indices` 列出三维网格每个位置的整数坐标，形状是 [3,2,2,2]。
- `reshape(3,-1)` 把后三个维度展开，形状变成 [3,8]；−1 表示由元素总数自动推断。
- `.T` 转置二维数组，得到 [8,3]，每一行是一块的 `(ix,iy,iz)`。

结果顺序为：

```text
节点0 → (0,0,0)
节点1 → (0,0,1)
节点2 → (0,1,0)
节点3 → (0,1,1)
节点4 → (1,0,0)
节点5 → (1,0,1)
节点6 → (1,1,0)
节点7 → (1,1,1)
```

z 的编号变化最快。后面的 ids 使用同一顺序，这保证中心数组和图边引用的是同一批节点。

## 8. column_stack：形成每个单元的上下边界

```python
cell_lower = np.column_stack([axes[d][ijk[:, d]] for d in range(3)])
cell_upper = np.column_stack([axes[d][ijk[:, d] + 1] for d in range(3)])
```

`ijk[:,d]` 取所有节点在第 d 个方向的整数编号。把这些编号放进 `axes[d][...]`，得到对应的左侧边界。编号加 1 就得到右侧边界。

`column_stack` 把 x、y、z 三列并成 [N,3] 的数组。随后：

```python
widths = cell_upper - cell_lower
centers = (cell_lower + cell_upper) / 2
```

都是逐元素运算。widths 每一行存一块材料的三个边长；centers 每一行存一个高斯中心。

## 9. covariance：高斯宽度单独定义

```python
covariance = np.zeros((count, 3, 3), dtype=np.float64)
for d in range(3):
    covariance[:, d, d] = (sigma_ratio * widths[:, d]) ** 2
```

`zeros` 先分配全零数组。每个节点都有一个 3×3 矩阵，因此形状为 [N,3,3]。

这里只向对角线写入 σx²、σy²、σz²，没有旋转，因此高斯轴与世界坐标轴对齐。协方差的单位为 m²。标准差和单元宽度都是正数，所以矩阵正定。

这不是渗透率张量，也不包含木纤维方向信息。以后若添加材料方向，要独立定义。

## 10. volume 和孔隙容量：本阶段最核心的三行

```python
volume = np.prod(widths, axis=1)
pore_volume = volume * porosity
capacity_mass = water_density * pore_volume
```

`np.prod(...,axis=1)` 沿每一行把三个边长相乘，得到 [N] 的体积数组。`axis=1` 是“对列方向归约”；如果省略，可能把所有节点的所有边长乘成一个数，那就错了。

对于 4×4×4 示例：

```text
单元边长 = 0.025 × 0.015 × 0.010 m
单元体积 = 0.00000375 m³ = 3.75 cm³
单元孔隙体积 = 3.75 × 0.6 = 2.25 cm³
单元最大水质量 = 2.25 g
64 个单元 → 144 g
```

如果三个方向的划分数都翻倍，节点数变成 8 倍，每个节点的代表体积变成 1/8，所以总容量不变。

这份实现使用“域划分分配体积”，没有使用“归一化高斯核积分分配体积”。两者都是计划书提出的候选方向；这里选择前者作为容易严格检查的基线。

## 11. chunks、append、concatenate 是为了什么

build_model 允许输入多个独立 Box。每个盒体先产生自己的一批数组，保存到 `chunks` 字典的列表里。

- 字典 `chunks['centers']`：按名字找到中心数组列表。
- 列表 `.append(centers)`：把当前盒体的一整批中心加入列表，不改变中心数值。
- 最后的 `np.concatenate(parts,axis=0)`：沿节点方向把各批接起来。

多个盒体的局部节点编号都从 0 开始，因此图边要加 `offset` 变成全局编号。第一个盒体有 64 个点时，第二个盒体的局部 0 号应变成全局 64 号。

`np.full(count, box.porosity)` 创建 count 个相同孔隙率值；它不是调用物理公式。第一版每个盒体内设均匀参数。

## 12. 连通图：为什么不用最近邻搜索

本阶段知道精确的单元划分，因此直接利用共享面关系。仅仅位置接近或高斯核重叠并不足以证明中间有材料。

```python
ids = np.arange(count).reshape(tuple(shape))
```

`arange(count)` 创建 0 到 count−1 的编号，再排成三维布局。现在 `ids[ix,iy,iz]` 就告诉我们该单元在平坦数组中的编号。

```python
left, right = [slice(None)] * 3, [slice(None)] * 3
left[d], right[d] = slice(0, -1), slice(1, None)
i = ids[tuple(left)].ravel()
j = ids[tuple(right)].ravel()
```

`slice(None)` 等价于取这一轴的全部元素；`slice(0,-1)` 取除最后一个以外的元素；`slice(1,None)` 取从第二个到末尾的元素。

例如沿 x 方向，一边取 x=0..nx−2，另一边取 x=1..nx−1，就配成所有相邻单元。`ravel()` 将结果展开成一维。

`np.column_stack((i,j))` 把对应的起点与终点组合成 [E,2] 的边表。三次循环分别生成 +x、+y、+z 的边；反方向不再存一次，以免重复。

无向不意味着只能正方向流动。B 阶段可以在一条边上保存有符号通量，同时更新两个端点。

## 13. 每条边为什么还保存面积、距离和法线

如果沿 x 方向相邻，共享面面积为 dy·dz，中心距离为 dx，方向为 (1,0,0)。另外两个方向类似。

代码中 `other_axes` 选出除 d 以外的两个方向，`np.prod` 将它们的边长相乘。法线用一个三维零向量在第 d 位写 1 得到。

这些量将来可用于规则正交网格上的通量几何因子 A/d。但当前没有材料渗透率和压力差，所以**边面积除以距离并不是完整水力导通系数**。对任意非对角各向异性张量，也不能不加分析地使用简单两点公式。

## 14. 空气间隙是怎样处理的

演示中左域 x∈[0,0.045]，右域 x∈[0.055,0.10]。中间的 0.01 m 没有任何材料单元，因此没有分配材料体积。

每个盒体只在自己内部生成边，跨域边不会出现。这是显式材料域规则，不是算法从图像里自动发现了裂缝。

两个盒体即使贴在一起也不会连边。此限制必须牢记：此版本支持独立材料域，尚未实现跨盒体的连通材料界面。

## 15. InternalGaussianModel 与 @property

这个 dataclass 把构建结果集中存起来。比如 `model.centers` 是坐标数组，`model.edges` 是边表。

`pore_volume`、`capacity_mass`、`water_mass` 用 `@property` 定义。访问 `model.capacity_mass` 时，Python 实际调用相应函数并返回计算结果，但外观上像读一个普通属性。

这样不用同时保存 volume、pore_volume、capacity_mass 三份可能互相矛盾的独立状态。当前 `water_mass` 由初始 saturation 推导；到 B 阶段应设计质量为主状态的更新方式，不能修改一次临时返回的 water_mass 数组后就认为模型中的饱和度也变了。

## 16. component_labels：检查是不是连成一块

它先用边表构建邻接列表；每一条无向边 i—j 同时记录 i 的邻居 j 和 j 的邻居 i。

labels 初始全为 −1，代表未访问。选一个未访问点放进 stack，反复取出它，把尚未访问的邻居标为相同分量并放入 stack。直到这一块全部遍历完，再从下一个未访问点开始。

`stack.append` 入栈，`stack.pop` 取出最近放入的元素，这叫深度优先遍历。其目的只是检查拓扑，不是在模拟水流。

完整块体应只有一个分量，空气间隙示例应有两个；单个孤立节点也算一个分量。

## 17. summary 和 save

`summary()` 用 `.sum()` 汇总所有节点的体积、容量和初始水量，转换成 Python 普通数字后可写进 JSON。容量之和在孔隙率不均匀时必须算 ΣρV_iφ_i，不能随意用一个未加权平均孔隙率替代。

`save()` 使用 NumPy 的 `savez_compressed`，把命名数组打包压缩为 NPZ。关键字 `centers=...` 就是以后用 `data['centers']` 读取的名称。`schema_version=1` 用于以后调整数据格式时区分版本。

`Path` 来自标准库 pathlib，负责路径组合；`mkdir(parents=True,exist_ok=True)` 在需要时创建父目录，已存在不报错。

## 18. run_demo 中其余函数

`argparse.ArgumentParser` 接收可选的 `--output` 路径。`for n in (4,8,16)` 用三个分辨率对同一盒体重复构建，并用 `assert` 在不满足条件时立即报错。

`np.testing.assert_allclose` 比较浮点数组在容差内是否相等；`assert_array_equal` 检查精确相等。协方差的缩放关系用前者，未改变的容量和边表用后者。

`csv.DictWriter` 把字典列表写为表格；`json.dumps` 将报告转换成文本；这些都不参与物理计算。

`write_slice_svg` 选出最底层高斯，把 x、y 映射到图像坐标。`np.isclose` 避免浮点数比较误差；`flatnonzero` 返回满足条件的索引。每个代表单元画矩形、每个高斯画 1σ 椭圆、同层边画线。

SVG 是文本矢量图，浏览器可以打开。图中椭圆来自本阶段构建的协方差，不是调用官方高斯渲染器，也不验证遮挡、透明度或照片重建效果。

## 19. 7 项测试各自在防什么错误

| 测试 | 防止的错误 |
|---|---|
| resolution_invariance | 增加高斯后复制了材料体积或孔隙容量；无效协方差或中心越界 |
| graph_faces_and_edge_count | 编号错位、重复边、面积/距离/方向错误 |
| capacity_independent_of_covariance | 错把高斯椭球大小当作物理容量 |
| air_gap_and_materials | 跨空气连边，或不同孔隙率的容量/初始水量累计错误 |
| zero_porosity_and_singleton | 无孔隙仍有容量，空边数组形状错误 |
| invalid_inputs | 负体积、非法分辨率、非法状态、重叠体积进入系统 |
| save_roundtrip | 保存后字段丢失或数值改变 |

## 20. 本阶段为什么达到 A 的目标，又没有证明什么

构建器给每个高斯分配了明确的中心、协方差和代表体积，也提供了不跨空气的材料图。数学上 ΣV_i 等于盒体体积；程序以三个分辨率和多项测试检验了这一点。固定孔隙率时 ΣρV_iφ_i 随之不变。

但该结论不涉及水传播速度、真实材料参数、连续含水场查询或湿润光学。没有动态分裂/合并，因此不同分辨率实验是从同一初始状态重新构建，而不是已经实现了带水状态的守恒重采样。

下一步 B 可以在此输出上写成对质量通量，当前无需修改官方 3DGS 的 train.py 或 CUDA 光栅化器。
