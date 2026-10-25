"""Generate static figures and RESULTS.md exclusively from saved experiment data."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from partition import Nodes


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=Path(__file__).parent/'outputs')
    out = p.parse_args().output
    report = json.loads((out/'report.json').read_text(encoding='utf-8'))
    if report['status'] != 'PASS':
        raise RuntimeError('Cannot publish successful figures from incomplete/failed run')
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for d in (1, 2, 3):
        data = [r for r in report['convergence'] if r['case'] == f'perturbed_{d}d']
        ax.loglog([r['cells'] for r in data], [r['volume_relative_l1'] for r in data], 'o-', label=f'{d}D')
    ax.set(xlabel='Midpoint cells per axis', ylabel='Relative L1 node-volume error',
           title='Quadrature convergence (fixed Gaussian nodes)')
    ax.set_xticks([8, 16, 32], labels=['8', '16', '32'])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which='both', alpha=0.25); ax.legend()
    fig.savefig(out/'convergence.png', dpi=170); fig.savefig(out/'convergence.svg'); plt.close(fig)

    data = np.load(out/'nonuniform_nodes_reference.npz')
    nodes = Nodes(data['centers'], data['covariance'], data['labels'])
    x = np.linspace(0, 0.1, 600)
    _, w, _ = nodes.weights(x[:, None], 0)
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True)
    axes[0].stackplot(x*1000, w.T, labels=[str(i) for i in range(len(nodes.centers))], alpha=0.85)
    axes[0].set(xlabel='Position (mm)', ylabel='Volume ownership fraction', ylim=(0, 1), title='Nonuniform Gaussian partition: fractions sum to one')
    axes[1].bar(np.arange(len(nodes.centers)), data['volume']*1e9)
    axes[1].set(xlabel='Gaussian node index', ylabel='Representative volume (mm³)', title='Integrated volume, not Gaussian ellipsoid volume')
    fig.savefig(out/'partition.png', dpi=170); fig.savefig(out/'partition.svg'); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, sigma in zip(axes, (0.6, 1.2)):
        data = np.load(out/f'regular_4_sigma_{sigma}_comparison.npz')
        ratio = (data['gaussian_volume']/data['inherited_volume']).reshape(4, 4, 4)[:, :, 0]
        im = ax.imshow(ratio.T, origin='lower', cmap='coolwarm', vmin=0.6, vmax=1.4)
        ax.set(title=f'Boundary slice, sigma / cell width = {sigma}', xlabel='x node', ylabel='y node')
        for i in range(4):
            for j in range(4):
                ax.text(i, j, f'{ratio[i,j]:.3f}', ha='center', va='center', fontsize=9)
    fig.colorbar(im, ax=axes, label='Gaussian volume / inherited cell volume')
    fig.savefig(out/'inherited_comparison.png', dpi=170); fig.savefig(out/'inherited_comparison.svg'); plt.close(fig)

    cases = report['cases']; large = report['large_scale']
    lines = ['# M2 实验结果（方案 A）', '', '执行日期：2026-10-21。状态：**PASS（本轮 M2 范围）**。', '',
             f'共 {len(cases)} 个通用积分实验，另有 4 级大规模可分离高斯精算。总实验耗时 {report["elapsed_seconds"]:.2f} s。',
             '数字由 outputs/report.json 自动生成；单位测试另见 outputs/unit_tests.txt。', '',
             '## 体积与容量账本', '',
             f'- 最大分区和误差：{max(r["unity_max_error"] for r in cases):.3e}。',
             f'- 最大体积分配总量相对误差：{max(r["partition_total_relative_error"] for r in cases):.3e}。',
             f'- 最大求积域体积相对解析体积误差：{max(r["domain_quadrature_relative_error"] for r in cases):.3e}。',
             f'- 最大孔隙容量分配相对误差：{max(r["capacity_partition_relative_error"] for r in cases):.3e}。',
             '- 所有实验节点体积为正、容量非负；零孔隙率实验容量为零。',
             '- 8/64/512 节点与核宽比例 0.3/0.6/1.2 的均匀容量扫描通过；非均匀材料 4/8/16 节点与 0.3/1.2 核宽扫描通过。',
             '- 分离域独立恢复各自体积与容量，即使高斯尾部明显重叠。', '',
             '## 局部积分收敛', '',
             '误差是 sum_i |V_i - V_i_ref| / sum_i |V_i_ref|，不是仅检查总体积。', '',
             '| 布局 | 8 格/轴 | 16 格/轴 | 32 格/轴 | 加密参考之间差异 |',
             '|---|---:|---:|---:|---:|']
    for d in (1, 2, 3):
        errors = [r['volume_relative_l1'] for r in report['convergence'] if r['case'] == f'perturbed_{d}d']
        stability = next(r['volume_relative_l1'] for r in report['convergence'] if r['case'] == f'perturbed_{d}d_reference_stability')
        lines.append(f'| {d}D 扰动 | {errors[0]:.6e} | {errors[1]:.6e} | {errors[2]:.6e} | {stability:.3e} |')
    lines += ['', '二维另含旋转椭圆核；上述结果验证固定节点下的**求积收敛**，不是输运方程的空间或时间收敛。', '',
              '![求积收敛](outputs/convergence.png)', '',
              '## 6912 节点验证', '',
              f'- 通用全节点中点求积与同规则因子化结果差异：{large["factorization_relative_error"]:.3e}。',
              f'- 通用粗中点积分相对高精度体积分配的 L1 误差：{100*large["generic_midpoint_relative_l1_vs_reference"]:.6f}%。',
              f'- 96 与 192 分段的四点 Gauss–Legendre 参考差异：{large["reference_check_relative_l1"]:.3e}。',
              '- 高精度加速依赖规则乘积布局的严格可分离性，不代表一般非规则 6912 节点性能。', '',
              '## 与继承方格体积的差别', '',
              '| 配置 | 体积差异相对 L1 | 最大单节点相对变化 |', '|---|---:|---:|']
    for r in report['inherited_comparison']:
        lines.append(f'| {r["case"]} | {r["volume_relative_l1_vs_inherited"]:.6e} | {r["max_relative_node_change"]:.6e} |')
    lines += ['', '这是两种体积定义的差异，不是“高斯误差”；总体积相同不要求每个节点体积等于旧方格。', '',
              '![体积比](outputs/inherited_comparison.png)', '',
              '## 非均匀材料与诊断', '',
              f'- 本例仅用中心孔隙率乘节点体积，总容量偏差为 {100*report["center_sample_phi_capacity_relative_error"]:.4f}%；一致容量积分恢复解析总容量。',
              '- 远离所有节点的位置仍能稳定归一化，但记录了巨大覆盖距离，不能据此认为核覆盖合理。',
              '- 缺少域支持、非法孔隙率、无效协方差、重叠积分域等由测试验证为显式拒绝。', '',
              '![非均匀节点分区](outputs/partition.png)', '',
              '## 验收范围与尚未完成', '',
              '- 已完成：本目录 README 所列 M2 方案 A 实验与对照。',
              '- 仍假设：物体域、连通标签及孔隙率场已知；核宽由受控实验指定，不是自动最优选择。',
              '- 未完成：旧 FVM 严格等价对照（源码缺失）；一般复杂几何；KD-tree/截断；M3 面与通量；M4 新几何输运。',
              '- 本轮验收未发生失败；若后续运行失败，脚本保存带时间戳的 failure 日志，将 report 状态标为 FAIL，不沿用旧 PASS。',
              '- 不宣称输运精度或性能优于 FVM；未测峰值内存，耗时仅为本机实验记录。', '',
              '原始数据：outputs/metrics.csv、convergence.csv、inherited_comparison.csv、report.json 及对应 NPZ。']
    (out.parent/'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('Generated 3 PNG/SVG figures and RESULTS.md')


if __name__ == '__main__':
    main()
