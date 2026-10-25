# M2 文献依据与采用范围

阅读日期：2026-10-21。依据本次对话已核对的原文；不以摘要冒充全文阅读。

| 资料 | 读取位置 | 原文支持 | 本项目采用 / 限制 |
|---|---|---|---|
| Hopkins (2015), A New Class of Accurate, Mesh-Free Hydrodynamic Simulation Methods | arXiv v2 PDF 第 5–6 页 §2.1，式 (6)–(10)；第 10 页 §2.7，式 (27) | 归一化核分区；有效体积为分区积分；体积求积近似及代价 | 借鉴分区思想，使用本项目独立定义的 a_i=1 高斯及有限物体域积分；不搬用 1/omega 近似或 Riemann 求解器，不迁移其紧支撑核精度结论 |
| Eirís et al. (2023), MLS-SPH-ALE | HTML §1.1–1.2、§4.2 式 (25)、(30)–(31)，§5.2 式 (66) | 几何体积与有效体积的区别；PU 导出总量恒等式；FVPM 粒子函数 | 作为 M2 数学组织依据；仍需数值求积验证，PU 不独立保证线性场重建或输运一致性 |
| Foy, Perré, Turner (2017), The Meshfree Finite Volume Method with application to multi-phase porous media models | 出版社摘要及公开正文片段；未取得全文 | 控制体结合插值通量，面向多相多孔介质 | 只确认替代路线背景；未实现、未借用未核实公式；不是高斯分区公式来源 |
| Quinlan, Lobovský, Nestor (2014), Development of the meshless finite volume particle method with exact and efficient calculation of interparticle area | 作者机构摘要；未取得全文 | top-hat 核可用于精确高效的粒子交换面积计算 | 说明积分代价与核选择相关；不能据此给平滑高斯宣称解析交换面积 |

原文链接：

- https://arxiv.org/pdf/1409.7395
- https://link.springer.com/article/10.1007/s11831-023-09965-2
- https://www.sciencedirect.com/science/article/abs/pii/S0021999116307094
- https://research.universityofgalway.ie/en/publications/development-of-the-meshless-finite-volume-particle-method-with-ex-6/

## 本项目推导

对同一连通域，令 psi_i=G_i/sum_j G_j。采用共享正权重求积 (x_k,w_k)，
V_i^Q=sum_k w_k psi_i(x_k)，则 sum_i V_i^Q=sum_k w_k（浮点误差除外）。
同理 C_i^Q=rho sum_k w_k psi_i(x_k) phi(x_k)，其和恢复同一求积下的孔隙容量。
非负权重且 0<=phi<=1 给出 0<=C_i<=rho V_i。
该恒等式不控制单节点求积误差，故实验另设独立积分加密与参考检查。

规则乘积节点和轴向可分离核下，G_ijk=g_i(x)g_j(y)g_k(z)，
sum_ijk G_ijk=(sum_i g_i)(sum_j g_j)(sum_k g_k)，
因此 psi_ijk=psi_i(x)psi_j(y)psi_k(z)。乘积域和乘积求积允许三维 V_ijk
精确因子化为三个一维分区积分。此处“精确”指同一求积规则的代数等价，
不是声称有限求积无误差，也不是一般非规则节点加速定理。
