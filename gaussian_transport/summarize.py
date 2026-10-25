"""Generate M3--M6 reports from the successful raw report.json."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; p=json.loads((ROOT/'gaussian_transport/outputs/report.json').read_text(encoding='utf-8'))
if p['status']!='PASS_WITH_REJECTED_CANDIDATES': raise RuntimeError('successful raw run required')
reports=ROOT/'reports'; reports.mkdir(exist_ok=True)
m2_path=ROOT/'stage_m2_partition/outputs/report.json'
m2_data=json.loads(m2_path.read_text(encoding='utf-8')) if m2_path.exists() else None

def save(name,text): (reports/name).write_text(text.strip()+'\n',encoding='utf-8')

save('00_repo_audit.md',f'''# 仓库审计

## 已实现

| 范围 | 实际入口 | 状态与单位 |
|---|---|---|
| Stage A | `stage_a_internal/model.py` | 规则盒体节点、继承单元体积和六邻接；m、m³、kg |
| Stage B | `stage_b_transport/solver.py` | 规则正交图上的质量主状态、毛细/重力、三库存和限流器 |
| M2 | `stage_m2_partition/partition.py` | 高斯 PU 体积/容量积分，已验证 |
| M3--M6 | `gaussian_transport/` | 独立 CPU 原型，本报告所述 |

旧端到端 FVM 原型源码不在仓库中，因此 M0 的原始基线复跑和 M1 的逐状态严格等价为 **BLOCKED**。本轮没有把 Stage B 或方案文字中的历史数值冒充旧源码重跑结果。

连续物理单位：固有渗透率 κ 为 m²，动力黏度 μ 为 Pa·s，水力势 H 为 Pa，质量通量 F 为 kg/s。主状态 m 为 kg，饱和度 S=m/C。表面状态是有质量账本的 reservoir，不代表解析水膜横向流动。

本轮 Git HEAD `{p['git_commit']}` 只标识已提交基线；未提交的新代码应以工作区内容为准。随机种子 {p['seed']}。
''')

save('01_literature_matrix.md','''# 文献—公式—适用性矩阵

| 来源 | 原文支持 | 本项目采用 | 不直接迁移 |
|---|---|---|---|
| Hopkins 2015, §2.1/§2.7 | PU 核分区、有效体积、矩阵梯度 | `ψ_i` 与 `V_i=∫ψ_i` 的组织方式 | 天体流体 Riemann 求解器、特定紧支撑核和移动粒子结论 |
| Eirís et al. 2023, §4.2/§5 | meshless-FV 有效体积和成对有效面积结构 | `A_ij=∫(ψ_i∇ψ_j-ψ_j∇ψ_i)`，标为候选 | 不宣称已证明适用于高斯 Richards 两点通量 |
| Foy et al. 2017 | 不重叠控制体与插值通量的多孔介质路线 | 作为混合控制体路线背景 | 未获得全文的公式不作实现依据 |
| Quinlan et al. 2014 | top-hat 核下粒子交换面积可精确高效计算 | 作为核选择/积分成本警示 | top-hat 精确面积不能套到平滑高斯 |
| MRST MPFA 文档 | TPFA 在非 K 正交/各向异性时不一致；MPFA 用多点 | 用线性 patch test决定 TPFA适用范围；实现 LS 梯度重构诊断 | 未把 MRST 实现或 MPFA-O 名称移植到本项目 |

公式分级：裁剪 Voronoi/标准 TPFA 属于已有有限体积方法；PU 有效面结构有文献推导；高斯有效面用于 Darcy 的投影两点通量及全局 PU 重映射属于**本项目候选**；中心/协方差采样规则属于实验设计。
''')

ra=p['route_a']; rb=p['route_b']
m2=m2_path
m2_note = (f"M2 原始汇总位于 `{m2.relative_to(ROOT)}`。" if m2.exists()
           else "M2 的代码和单元测试已存在；运行 `python -B stage_m2_partition/run_experiments.py` 可重新生成原始汇总。")
save('02_gaussian_volume_partition.md',f'''# M2：高斯体积分区

M2 使用归一化高斯权重 `ψ_i=G_i/Σ_jG_j` 构成单位分解，并用共享的正权重 Gauss--Legendre 求积计算 `V_i=∫ψ_i`、`∫ψ_iφ` 和容量 `C_i=ρ∫ψ_iφ`。所有节点使用同一组积分点与权重，因此离散层面有 `Σ_iV_i=|Ω|`；非均匀孔隙率通过 `φ_i^eff=∫ψ_iφ/V_i` 聚合。

中心 `μ_i` 是节点的空间位置。协方差采用 `Σ_i=R_i diag(σ_1²,…,σ_d²)R_i^T`，必须对称正定；它控制高斯核覆盖范围和方向，不直接充当孔隙体积或渗透率。M2 的规则、扰动、边界和非均匀材料实验已经验证总体积、容量和非负性，同时也显示核宽会改变局部 `V_i`：全局守恒不能证明局部分配正确。

{m2_note} 详细解释与限制见 `stage_m2_partition/RESULTS.md`。
''')

save('03_neighbor_and_effective_face.md',f'''# M3：邻接和有效面

## 路线 A：裁剪 Voronoi（通过，主路线）

对每个中心，用所有垂直平分面裁剪物体域，得到不重叠控制体；共享平分面给出邻接、面积、法向、面心和中心距离。矩形总体积误差 {ra['box_volume_error']:.3e}，边界长度误差 {ra['box_boundary_error']:.3e}；L 形凹域体积误差 {ra['l_volume_error']:.3e}，边界误差 {ra['l_boundary_error']:.3e}。不同物体通过显式分量合并，跨分量边为 {p['disconnected']['cross_component_edges']}。

这是“高斯节点 + 非结构有限体积”的混合方法。高斯核可用于状态重建，但输运体积和面来自显式控制体。

## 路线 B：平滑高斯有效面（几何通过，通用两点通量拒绝）

用共享 Gauss--Legendre 求积计算 `V_i`、`A_ij` 和边界 `B_i`。反对称残差 {rb['anisotropic_kernel']['antisymmetry_max']:.3e}，几何闭合相对残差 {rb['anisotropic_kernel']['closure_relative']:.3e}，有限差分梯度误差 {rb['gradient_fd_error']:.3e}。

但是随机旋转各向异性核中，将 `A_ij` 投影到中心线再做两点通量，线性 patch 相对误差 {rb['anisotropic_kernel_projected_tpfa_patch_error']:.3f}。因此保留完整面积向量和失败结果，不把投影标量面积接入主非线性求解器。

32×32、三点求积与 48×48、四点参考之间，体积向量相对 L2 差 {rb['quadrature_crosscheck']['volume_relative_l2']:.3e}，面积向量差 {rb['quadrature_crosscheck']['area_vector_relative_l2']:.3e}，边界向量差 {rb['quadrature_crosscheck']['boundary_vector_relative_l2']:.3e}；边界标量总长度误差 {rb['boundary_perimeter_error']:.3e}。剪枝消融写入 `gaussian_transport/outputs/route_b_pruning.csv`：阈值提高时边数减少，但闭合残差上升，所以主结果保留全对积分，不用任意搜索半径伪造稀疏性。
''')

ledger=p['ledger']
save('04_transport_and_mass_ledger.md',f'''# M4：通量与水质量账本

主输运使用路线 A。各边法向渗透率先在两端计算 `n^T κ n`，再调和平均；标量/K 正交情况下：

`T_ij = ρ κ_h A_ij /(μ d_ij)`，`F_ij=T_ij k_r,ij(H_i-H_j)`。

每条无向边只计算一次交换量；总流入/流出限流器同时受供体库存和受体剩余容量约束，并向两端施加同一个缩放量。没有更新后裁剪。

开放边界使用 free、surface、pore 三份质量库存。湿润实验最大账本残差 {ledger['wet']['max_mass_residual_kg']:.3e} kg；不透水实验孔隙水 {ledger['impermeable']['pore']:.3e} kg，供水保留在 surface/free，最大残差 {ledger['impermeable']['max_mass_residual_kg']:.3e} kg。

实现顺序覆盖线性毛细、van Genuchten 型非线性响应和重力；这些参数仍是合成值，不是材料标定结果。
''')

conv=p['convergence']; tc=p['time_convergence']; patch=p['patch_tests']
rows='\n'.join(f"| {r['layout']} | {r['n']} | {r['l2']:.6e} | {r['front_error_m']:.3e} | {r['mass_residual_kg']:.3e} | {r['preprocess_s']:.4f} | {r['solve_s']:.4f} | {r['preprocess_peak_bytes']} |" for r in conv)
trows='\n'.join(f"| {r['dt_s']:.6g} | {r['l2_vs_dt_0_00625']:.6e} | {r['mass_residual_kg']:.3e} |" for r in tc)
save('05_validation_and_baselines.md',f'''# M5：一致性、收敛和基线

## 空间收敛（解析零通量余弦扩散）

| 布局 | N | 饱和度 L2 | 等值面位置误差 m | 质量残差 kg | 构图 s | 推进 s | Python 峰值 bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
{rows}

规则布局在 N=16 已低于 1e-5；相同自由度下扰动布局误差明显更大。以 L2≤5e-4 为目标，本组规则布局 N=16 达到，扰动布局需 N=64。这是当前受控问题的结果，不推广为普遍性能排名。

## 时间误差（N=32，与 dt=0.00625 s 参考比较）

| dt s | L2 | 质量残差 kg |
|---:|---:|---:|
{trows}

误差随 dt 减半下降，空间和时间误差已分开统计。

常势最大内部流率 {patch['constant_potential_max_rate']:.3e} kg/s；各向同性 Voronoi TPFA 线性 patch 误差 {patch['isotropic_tpfa_relative_error']:.3e}。各向异性 TPFA 误差 {patch['anisotropic_tpfa_relative_error']:.3f}，多点最小二乘梯度重构的同一线性测试误差 {patch['anisotropic_reconstructed_relative_error']:.3e}。表中的“等值面位置”用解析余弦场 `S=0.5` 的交点 `x=L/2` 定义，用来提供方案要求的前沿位置误差；它不是尖锐湿润前沿。

旧外部 FVM 源码缺失，严格旧基线仍 BLOCKED。规则 Voronoi在一维退化为标准单元中心 FVM，可作为内部解析基准，但不冒充外部旧实现。
''')

d3=p['three_d']; ad=p['adaptive_remap']
save('06_failure_cases_and_next_questions.md',f'''# M6 扩展、失败案例和边界

已验证 L 形凹域、分离物体、固定 27 节点扰动三维盒体、各向异性渗透率 patch test，以及全局 PU 分裂/合并重映射。

三维裁剪 Voronoi和高斯 PU 都恢复总体积，但逐节点体积定义相对 L1 差 {d3['volume_definition_relative_l1']:.4f}；这说明守恒不能消除方法定义差异。3D Voronoi 构图耗时 {d3['voronoi_preprocess_s']:.3f} s，Python 跟踪峰值 {d3['voronoi_peak_bytes']} bytes。

全局分裂重映射质量相对误差 {ad['split']['relative_mass_error']:.3e}，合并误差 {ad['merge']['relative_mass_error']:.3e}；但分裂再合并的饱和度 L2 为 {ad['roundtrip_saturation_l2']:.4f}，显示守恒重映射仍引入平滑/扩散。

保留的失败：平滑高斯完整面积向量在方向性核下通常不平行中心线，强行投影做 TPFA 的 patch test 失败；一般各向异性 Voronoi TPFA 也失败。主求解器因此只允许标量/K 正交 TPFA。最小二乘梯度重构仅完成线性诊断，尚不是完整单调 MPFA 非线性求解器。

仍需外部条件：旧 FVM 源码、真实复杂物体的可靠分割/连通数据、材料参数和独立实验观测。CUDA、湿润渲染、形变、蒸发和参数反演不属于本计划 A/B 范围。
''')

save('COMPLETE_SUMMARY.md',f'''# 体积高斯守恒输运完整总结

## 变量与计算方法

| 变量 | 计算方法 | 最终用途/结论 |
|---|---|---|
| `μ_i` | 规则中心、受控扰动或域内指定采样点 | 高斯/控制体节点位置，m |
| `Σ_i` | `R diag(σ_d²) R^T`，SPD 检查 | 只控制核覆盖和状态重建，不等于 κ |
| `G_i` | `exp[-0.5(x-μ_i)^TΣ_i^-1(x-μ_i)]` | 未归一化高斯核 |
| `ψ_i` | 同连通域内 `G_i/Σ_jG_j` | 非负 PU 体积分区 |
| `V_i^G` | 共享正权重 Gauss--Legendre 积分 `∫ψ_i` | 高斯有效体积；M2/重建使用 |
| `φ_i^eff` | `∫ψ_iφ / V_i^G` | 非均匀孔隙率一致聚合 |
| `C_i` | `ρ∫ψ_iφ`（高斯路线）或 `ρφ_iV_i`（控制体路线） | 最大孔隙水质量 kg |
| `V_i^CV` | 物体域裁剪 Voronoi 单元面积/体积 | 主输运的不重叠控制体 |
| `N(i)` | 具有正共享 Voronoi 面的同分量节点；分量间不连边 | 主输运邻接 |
| `A_ij,n_ij,d_ij` | 裁剪 Voronoi公共面、从 i 到 j 法向、中心距 | 路线 A TPFA 几何 |
| `A_ij^eff` | `∫(ψ_i∇ψ_j-ψ_j∇ψ_i)`；边界 `B_i=∫ψ_i n` | 路线 B 研究候选；几何闭合通过，通用投影 TPFA 被拒绝 |
| `κ_ij` | 两端 `n^Tκn` 的调和平均 | 只在标量/K 正交 TPFA 中使用 |
| `T_ij` | `ρ κ_ij A_ij/(μd_ij)` | kg/(s·Pa) 边导通 |
| `H_i` | `-p_c(S_i)-ρ g·μ_i`（常压空气基准） | 水力势 Pa |
| `F_ij` | `T_ij k_r,ij(H_i-H_j)`，每条无向边一次 | 反对称质量流率 kg/s |
| `m_i` | 主守恒状态；同步成对交换并总量限流 | 保证 `0≤m_i≤C_i` |
| `S_i` | `m_i/C_i`，零容量时为 0 | 派生含水状态 |
| `free/surface/pore` | 独立库存，同量转移 | 开放边界质量账本 |
| `∇H`（一般各向异性） | 邻域多点最小二乘线性重构 | 通过线性 patch；当前仅诊断，未包装成完整 MPFA |
| 重采样质量 | `∫ψ_new,j ψ_old,i / V_old,i` 构造转移矩阵 | 全局守恒，但往返会产生数值扩散 |

## 最终方法选择

体积高斯 PU 用于代表体积研究、材料容量聚合和连续状态接口。可靠主输运采用裁剪 Voronoi控制体，因此最终可用原型是 **Gaussian nodes + non-structured finite volume**。平滑高斯有效面保留为研究算子；它满足反对称/闭合，但在各向异性核下不能压成可靠的中心线两点面积。

## 关键实验结果

- M2 的 6912 节点规则三维实验中，通用中点积分相对高精度分区体积的 L1 误差为 {m2_data['large_scale']['generic_midpoint_relative_l1_vs_reference'] if m2_data else float('nan'):.3e}；仅用中心孔隙率估容量的非均匀材料对照偏差为 {m2_data['center_sample_phi_capacity_relative_error'] if m2_data else float('nan'):.3%}。
- M3 路线 A 的矩形体积/边界误差分别为 {ra['box_volume_error']:.3e}/{ra['box_boundary_error']:.3e}；L 形域分别为 {ra['l_volume_error']:.3e}/{ra['l_boundary_error']:.3e}。
- M3 路线 B 的反对称残差为 {rb['anisotropic_kernel']['antisymmetry_max']:.3e}，闭合相对残差为 {rb['anisotropic_kernel']['closure_relative']:.3e}，但投影两点通量线性误差为 {rb['anisotropic_kernel_projected_tpfa_patch_error']:.3f}，所以被拒绝进入主求解器。
- M4 湿润实验最大总账本残差为 {ledger['wet']['max_mass_residual_kg']:.3e} kg；常势内部通量为 {patch['constant_potential_max_rate']:.3e} kg/s。
- M5 规则节点 N=16/32/64 的 L2 误差依次为 {conv[0]['l2']:.3e}、{conv[1]['l2']:.3e}、{conv[2]['l2']:.3e}；扰动节点误差更大但随分辨率下降。各向异性 TPFA patch 误差 {patch['anisotropic_tpfa_relative_error']:.3f}，线性最小二乘重构误差 {patch['anisotropic_reconstructed_relative_error']:.3e}。
- M6 三维两种守恒体积定义的逐节点相对 L1 差为 {d3['volume_definition_relative_l1']:.3%}；分裂—合并往返饱和度 L2 差为 {ad['roundtrip_saturation_l2']:.4f}，说明守恒重映射仍会扩散状态。

## 代码与复现入口

| 模块 | 内容 |
|---|---|
| `stage_m2_partition/partition.py` | 高斯核、单位分解、体积/容量求积 |
| `gaussian_transport/geometry.py` | 2D 裁剪 Voronoi、断开域合并、纯 NumPy 3D 半空间参考 |
| `gaussian_transport/smooth_faces.py` | `∇ψ_i`、平滑高斯有效面积分和闭合诊断 |
| `gaussian_transport/transport.py` | 毛细/重力势、TPFA、三库存、总量限流和多点重构诊断 |
| `gaussian_transport/remap.py` | 分裂/合并后的全局守恒 PU 重映射 |
| `gaussian_transport/run_all.py` | M3--M6 全实验及 JSON/CSV 输出 |

复现顺序：`python -B -m unittest gaussian_transport.test_pipeline -v`，`python -B -m gaussian_transport.run_all`，`python -B -m gaussian_transport.summarize`。

## 状态

| 里程碑 | 状态 |
|---|---|
| M0 旧原型审计/复跑 | BLOCKED：源码未获得；仓库审计完成 |
| M1 与旧 FVM 严格等价 | BLOCKED/PARTIAL：节点图分层已有，外部逐项对照缺失 |
| M2 高斯体积/容量 | PASS |
| M3 两条几何路线 | PASS_WITH_REJECTED_CANDIDATE：A 通过；B 几何通过、通用 TPFA 拒绝 |
| M4 守恒推进 | PASS：路线 A 标量/K 正交范围 |
| M5 验证 | PASS：账本、容量、解析扩散、时空收敛、patch、断裂和成本 |
| M6 扩展 | PASS_WITH_LIMITATIONS：凹域、3D、各向异性诊断、分裂/合并研究完成 |

原始数据在 `gaussian_transport/outputs/`；所有未通过候选均保留在报告，没有用质量守恒代替精度证明。
''')

print('Generated reports 00--06 and COMPLETE_SUMMARY.md')
