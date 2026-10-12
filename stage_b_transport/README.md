# 阶段 B：守恒水分输运与研究假设验证

本目录实现计划书中的 B 阶段 CPU 原型：节点水质量、毛细与重力通量、表面储水交换，以及与高斯表示相关的研究假设实验。复用 `../stage_a_internal/model.py`，不改动上游代码。仅需 NumPy。

## 先读方案，再运行

- [PLAN.md](PLAN.md)：B1–B4 实施顺序、公式、假设、对照和判据。
- [CODE_GUIDE.md](CODE_GUIDE.md)：主状态、通量、守恒限制器、自适应步长和研究模块的解释。
- [RESULTS.md](RESULTS.md)：运行后整理的数值结果及局限。

在项目根目录 PowerShell 中执行：

```powershell
python -m unittest discover -s stage_b_transport -p test_solver.py -v
python stage_b_transport/run_experiments.py
python stage_b_transport/run_followup.py
python stage_b_transport/summarize_outputs.py
```

全部结果写入本目录的 outputs。同名结果会被覆盖。使用普通有 NumPy 的 Python 环境也可以；无需 NVIDIA GPU、Warp 或 CUDA。

## 文件分工

| 文件 | 作用 |
|---|---|
| geometry.py | 阶段 A 适配、删除空气单元、提取顶部/底部储水面 |
| solver.py | 材料关系、三个质量库存、Darcy 图通量、有界守恒交换、自适应时间推进 |
| research.py | 新旧单元交叠重映射；体积加权、域约束的高斯含水查询 |
| run_experiments.py | 解析解、表面供水、重力、步长、裂缝、各向异性和研究假设对照 |
| run_followup.py | 在第一轮报告基础上追加严格容差、完全干燥初态与正则化敏感性实验 |
| summarize_outputs.py | 检查结果完整性、质量轨迹和饱和度范围，生成终态剖面对照图 |
| test_solver.py | 单元测试，包含故意错误的 clip 对照 |

## 输出

- `report.json`：所有实验的指标；只有 status=completed 才表示完整运行结束。
- `analytic_convergence.csv`、`analytic_*.npz`：解析解与空间收敛。
- `*_history.csv`：逐接受步的自由水、表面水、孔隙水、质量残差、饱和度和等效储水深度。
- `*_state.npz`：终态，用于后续阅读、绘图或阶段 C 接口。
- `*_inventory.svg`：实际数值轨迹生成的库存图，不是 GS 渲染视频。
- `remap.npz`、`query_refinement.csv`：候选方法与对照结果。

## 阶段定义

B1 的线性压力关系用于解析验证，不冒充真实材料。B2 使用正则化 VG/Mualem 型关系及合成参数；表面仅为储水库，外部供水是指定输入，不是 MPM 碰撞。B3 使用体素对齐裂缝与对角渗透率，不是任意曲面/任意张量算法。B4 是创新方向筛选，交叠守恒、核归一化等本身属于已有思想。

重要边界：S=1 时尚未增加独立压力未知量，不能处理通用饱和承压流；没有蒸发、渗出、双向液体反馈、曲面裁切、动态变形、WetSpongeCake 光学或真实材料定标。初步数值检查通过也不等于完成论文级物理验证。
