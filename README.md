# Gaussian Wetting

面向多孔材料吸水传播与高斯表示的研究原型。目前公开 **阶段 A：内部材料表示** 与 **阶段 B：守恒水分输运**，使用 Python / NumPy 在 CPU 上运行，无需 GPU、CUDA 或训练数据。

当前目标是先检查代表体积、储水容量、连通关系和数值输运是否正确。底层代表域仍以规则单元为基础，不宣称已实现任意非规则高斯输运、真实材料参数识别或完整湿润渲染。

## 当前进展

| 阶段 | 已实现 | 当前边界 |
|---|---|---|
| A 内部表示 | 内部高斯、非重叠代表体积、孔隙容量、域内邻接图、连通分量、分辨率对照 | 已知轴对齐盒体；不从照片恢复真实微孔 |
| B 守恒输运 | 节点水质量、毛细与重力通量、表面储水交换、三份水库存、有界守恒更新、自适应步长 | 静止骨架、指定参数和供水；尚未接入完整外部液体运动 |
| B 研究对照 | 交叠体积重映射、体积加权状态查询、阻隔/方向性和分辨率实验 | 候选方向验证，不构成已证明的新颖性或性能优势 |

## 目录与阅读顺序

```text
Gaussian-Wetting/
├── README.md
├── requirements.txt
├── stage_a_internal/
│   ├── model.py                 # 内部表示、代表体积与邻接图
│   ├── run_demo.py              # A 阶段演示与验收
│   ├── test_model.py
│   ├── CODE_GUIDE.md
│   └── outputs/                # 保存的示例结果
└── stage_b_transport/
    ├── geometry.py             # 材料域与边界适配
    ├── solver.py               # 守恒水分输运
    ├── research.py             # 重映射及状态查询
    ├── run_experiments.py
    ├── run_followup.py
    ├── summarize_outputs.py
    ├── test_solver.py
    ├── PLAN.md
    ├── CODE_GUIDE.md
    ├── RESULTS.md
    └── outputs/
```

建议先读 [A 阶段说明](stage_a_internal/README.md) 和 [A 代码导读](stage_a_internal/CODE_GUIDE.md)，再读 [B 实施方案](stage_b_transport/PLAN.md)、[B 代码导读](stage_b_transport/CODE_GUIDE.md) 与 [B 实验结果](stage_b_transport/RESULTS.md)。

## 安装

建议 Python 3.9 或更新版本。本地验证环境为 Python 3.9、NumPy 2.0.1。

```bash
git clone https://github.com/lixiangnew/Gaussian-Wetting.git
cd Gaussian-Wetting
python -m venv .venv
```

激活虚拟环境：Windows PowerShell 使用 `.\.venv\Scripts\Activate.ps1`；Linux/macOS 使用 `source .venv/bin/activate`。然后安装依赖：

```bash
python -m pip install -r requirements.txt
```

也可以直接使用已有的 NumPy 环境，无需创建虚拟环境。

## 运行与测试

以下命令均在仓库根目录运行。先执行不依赖已生成结果的单元测试：

```bash
python -m unittest discover -s stage_a_internal -p test_model.py -v
python -m unittest discover -s stage_b_transport -p test_solver.py -v
```

生成 A 演示，再按顺序运行 B 主实验、补充实验与结果汇总：

```bash
python stage_a_internal/run_demo.py
python stage_b_transport/run_experiments.py
python stage_b_transport/run_followup.py
python stage_b_transport/summarize_outputs.py
```

结果保存在对应阶段的 `outputs/` 中，重复运行会覆盖同名输出。B 完整实验比单元测试耗时更长；报告中 `status=completed` 表示该组实验执行结束。

## 物理状态与数据流

每个内部节点代表材料体积 V，孔隙率为 φ，饱和度为 S。水质量为 `m = ρw × V × φ × S`，最大储水质量为 `ρw × V × φ`。高斯协方差用于空间表示，不直接决定物理容量。

指定供水进入表面储水库，再与孔隙水交换；内部通过成对通量迁移水质量。一端扣除的质量与另一端增加的质量相同，并对可供水量和剩余容量施加约束。

自由水、表面水和孔隙水分别记账。表面等效厚度 h 与内部饱和度 S 分开保存；表面储水库不代表已模拟连续水膜的形状或横向流动。

## 已有验证与解释范围

- A：固定几何下改变节点数量，材料总体积与孔隙容量保持一致；检查核宽、空气间隙、零孔隙率等情况。
- B：检查质量守恒、容量界限、线性解析解、重力与渗透率作用，以及时间/空间离散敏感性。
- 研究对照：比较交叠守恒重映射与简单复制，以及体积加权查询与不合适的权重策略。

CSV/JSON 为数值指标；NPZ 为状态数组；SVG 为数值示意图。这些图不是 GS 渲染视频，合成参数结果也不是实测材料性能。具体数值和局限见各阶段报告。

## 尚未解决的问题

1. 复杂曲面、薄壁及裂缝的自动代表体积分配与连通性构建。
2. 一般非正交节点布局和任意各向异性张量上的准确输运离散。
3. 真实材料的孔隙率、渗透率及毛细曲线标定，以及非线性独立参考验证。
4. 饱和承压流、排水、表面横向流动和自由液体双向耦合。
5. 形变、湿胀、蒸发、困气，以及状态向完整湿润光学/GS 渲染的连接。

当前仓库不包含上游 3DGS / GaussianFluent 源码，也不需要它们才能运行 A/B。
