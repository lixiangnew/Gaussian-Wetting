# Gaussian + Voronoi transport

该目录实现高斯 + 裁剪 Voronoi 的守恒输运研究原型。主路线以内部高斯中心构造裁剪 Voronoi 控制体，用显式体积、公共交面和中心距离计算有限体积通量；归一化高斯用于连续查询、平滑有效面候选，以及节点拆分/合并后的守恒重映射。

总体方案和变量定义见仓库根目录的 [`高斯-Voronoi混合守恒输运方案.md`](../高斯-Voronoi混合守恒输运方案.md)。本目录依赖 `stage_m2_partition/partition.py` 提供高斯节点和稳定的单位分解权重。

## 文件

| 文件 | 内容 |
|---|---|
| `geometry.py` | 二维裁剪 Voronoi、L 形域和小规模三维盒域参考实现 |
| `transport.py` | 孔隙容量、渗透率、毛细/重力势、成对通量和有界质量更新 |
| `smooth_faces.py` | 基于高斯单位分解梯度的平滑有效面候选 |
| `remap.py` | 高斯拆分/合并后的守恒质量重映射 |
| `run_all.py` | 几何、输运、收敛、三维和重映射实验入口 |
| `test_pipeline.py` | 不依赖已有输出文件的单元测试 |
| `outputs/` | 一次完整运行生成的 JSON/CSV 结果 |

## 运行

从仓库根目录运行，核心代码只依赖 NumPy：

```powershell
python -B -m unittest discover -s stage_m2_partition -p test_partition.py -v
python -B -m unittest gaussian_transport.test_pipeline -v
python -B -m gaussian_transport.run_all
```

`summarize.py` 用于本地研究报告汇总，只有在同时保留 `reports/` 工作目录时才需要运行。原型不集成 CUDA 或完整 3DGS 渲染器。

## 解释范围

`geometry.py` 中的显式裁剪 Voronoi 是当前守恒主路线。`smooth_faces.py` 的平滑高斯有效面是研究候选，不能仅凭几何闭合就视为通过通量一致性验证。一般非正交、强各向异性问题需要多点通量或梯度重构。
