"""CPU 演示：分辨率改变、核宽改变、空气间隙三个独立检查。"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from model import Box, build_model


def write_slice_svg(model, path, title):
    """输出首个 z 层的二维剖面示意；椭圆是 1σ，不是孔隙边界。"""
    ids = np.flatnonzero(np.isclose(model.centers[:, 2], model.centers[:, 2].min()))
    lower, upper = model.cell_lower.min(axis=0), model.cell_upper.max(axis=0)
    scale = min(740 / (upper[0] - lower[0]), 400 / (upper[1] - lower[1]))
    def xy(p):
        return 50 + (p[0] - lower[0]) * scale, 480 - (p[1] - lower[1]) * scale
    lines = ['<svg xmlns="http://www.w3.org/2000/svg" width="860" height="550" viewBox="0 0 860 550">',
             '<rect width="860" height="550" fill="white"/>',
             f'<text x="30" y="30" font-family="sans-serif" font-size="18">{title}</text>',
             '<text x="30" y="55" font-family="sans-serif" font-size="13">Gray: representative cells; blue: 1-sigma Gaussian; orange: graph edges</text>']
    for i in ids:
        x, y = xy(model.cell_lower[i])
        w, h = (model.cell_upper[i] - model.cell_lower[i])[:2] * scale
        lines.append(f'<rect x="{x}" y="{y-h}" width="{w}" height="{h}" fill="none" stroke="#aaa"/>')
        x, y = xy(model.centers[i])
        rx, ry = np.sqrt(np.diag(model.covariance[i]))[:2] * scale
        lines.append(f'<ellipse cx="{x}" cy="{y}" rx="{rx}" ry="{ry}" fill="#1976d2" fill-opacity="0.12" stroke="#1976d2" stroke-width="0.6"/>')
    selected = set(ids.tolist())
    for i, j in model.edges:
        if i in selected and j in selected:
            x1, y1 = xy(model.centers[i]); x2, y2 = xy(model.centers[j])
            lines.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#df7b00" stroke-width="1.1"/>')
    lines.append('<text x="30" y="525" font-family="sans-serif" font-size="13">Schematic only: not a 3DGS render and not simulated water.</text></svg>')
    path.write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'outputs')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    # 同一个 10 × 6 × 4 cm 块体，同样的孔隙率 0.6。
    expected_volume = 0.10 * 0.06 * 0.04
    expected_capacity = 1000 * expected_volume * 0.6
    for n in (4, 8, 16):
        model = build_model([Box((0, 0, 0), (0.10, 0.06, 0.04), (n, n, n))])
        row = {'resolution': n, **model.summary()}
        row['volume_relative_error'] = abs(row['material_volume_m3'] / expected_volume - 1)
        row['capacity_relative_error'] = abs(row['capacity_kg'] / expected_capacity - 1)
        assert row['volume_relative_error'] < 1e-12
        assert row['capacity_relative_error'] < 1e-12
        assert row['components'] == 1
        rows.append(row)
        model.save(args.output / f'block_{n}.npz')
        write_slice_svg(model, args.output / f'block_{n}_slice.svg', f'Block: {n} x {n} x {n}, N={n**3}')
        print(f"N={n**3:4d}  V={row['material_volume_m3']:.9f} m3  capacity={row['capacity_kg']*1000:.6f} g  edges={row['edges']}")

    box = Box((0, 0, 0), (0.10, 0.06, 0.04), (8, 8, 8))
    narrow, wide = build_model([box], sigma_ratio=0.3), build_model([box], sigma_ratio=1.2)
    np.testing.assert_array_equal(narrow.capacity_mass, wide.capacity_mass)
    np.testing.assert_array_equal(narrow.edges, wide.edges)
    np.testing.assert_allclose(wide.covariance, narrow.covariance * 16)

    # 显式空气间隙 x∈(0.045, 0.055)，两侧材料域不会连边。
    gap = build_model([
        Box((0, 0, 0), (0.045, 0.06, 0.04), (4, 8, 8)),
        Box((0.055, 0, 0), (0.10, 0.06, 0.04), (4, 8, 8)),
    ], sigma_ratio=1.2)
    cross_edges = int(np.count_nonzero(gap.domain_id[gap.edges[:, 0]] != gap.domain_id[gap.edges[:, 1]]))
    assert cross_edges == 0 and gap.summary()['components'] == 2
    np.testing.assert_allclose(gap.volume.sum(), 0.09 * 0.06 * 0.04)
    gap.save(args.output / 'air_gap.npz')
    write_slice_svg(gap, args.output / 'air_gap_slice.svg', 'Two domains with an air gap; wide Gaussians do not create edges')
    with (args.output / 'resolution_check.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    report = {
        'resolution_check': rows,
        'kernel_width_check': {'sigma_ratios': [0.3, 1.2], 'capacity_unchanged': True, 'graph_unchanged': True},
        'air_gap_check': {**gap.summary(), 'cross_domain_edges': cross_edges},
        'scope': 'Stage A fixed box domains only. No transport, CUDA, or wet rendering.',
    }
    (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('PASS: resolution, covariance/capacity separation, and air-gap connectivity checks.')
    print('Outputs:', args.output.resolve())


if __name__ == '__main__':
    main()
