"""Reproducible M2-only experiments; default outputs stay in this directory."""
import argparse
import csv
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
import numpy as np
from partition import Box, Nodes, regular_nodes, integrate

SEED = 20261021
RUN_OUTPUT = None


def relative_l1(a, b):
    return float(np.sum(abs(a-b))/np.sum(abs(b)))


def check(r):
    m = r['metrics']
    assert m['unity_max_error'] < 1e-12
    assert m['partition_total_relative_error'] < 1e-10
    assert m['domain_quadrature_relative_error'] < 1e-10
    assert m['capacity_partition_relative_error'] < 1e-10
    assert np.all(r['volume'] > 0) and np.all(r['capacity'] >= 0)
    assert np.all(r['effective_porosity'] <= 1+1e-12)


def main():
    global RUN_OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent/'outputs')
    args = parser.parse_args(); out = args.output
    out.mkdir(parents=True, exist_ok=True)
    RUN_OUTPUT = out
    rng = np.random.default_rng(SEED)
    rows, comparisons, convergence = [], [], []
    started = time.perf_counter()
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    report = dict(status='RUNNING', seed=SEED, git_commit=commit,
                  python=platform.python_version(), numpy=np.__version__,
                  source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in Path(__file__).parent.glob('*.py')},
                  old_fvm_parity='BLOCKED: old FVM source unavailable; existing A/B is not an equivalence proof',
                  method='full Gaussian a_i=1; shared positive composite midpoint/Gauss-Legendre quadrature; no cutoff',
                  limits=['fixed box domains', 'no M3 faces or transport', 'no KD-tree/cutoff', 'no dynamic split/merge'])
    (out/'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')

    def run(name, nodes, boxes, cells, phi=0.6, order=1, analytic_capacity=None, batch=128):
        tick = time.perf_counter()
        r = integrate(nodes, boxes, cells, porosity=phi, order=order, batch_size=batch)
        check(r)
        row = dict(case=name, cells_per_axis=cells, order=order, batch_size=batch,
                   seconds=time.perf_counter()-tick, **r['metrics'])
        if analytic_capacity is not None:
            row['analytic_capacity_relative_error'] = abs(row['capacity_kg']-analytic_capacity)/analytic_capacity
            assert row['analytic_capacity_relative_error'] < 1e-10
        rows.append(row)
        (out/'partial_metrics.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        np.savez_compressed(out/(name+'.npz'), centers=nodes.centers, covariance=nodes.covariance,
                            labels=nodes.labels, **{k:v for k,v in r.items() if k != 'metrics'})
        print(name, 'N=', len(nodes.centers), 'Q=', row['quadrature_points'],
              'seconds=', round(row['seconds'], 3), flush=True)
        return r

    # Resolution and kernel-width invariance, with original equal-cell volumes.
    box = Box((0, 0, 0), (0.1, 0.06, 0.04))
    for n in (2, 4, 8):
        for sigma in (0.3, 0.6, 1.2):
            nodes = regular_nodes(box, n, sigma)
            name = f'regular_{n}_sigma_{sigma}'
            r = run(name, nodes, [box], 4*n, analytic_capacity=1000*0.6*box.volume)
            inherited = np.full(n**3, box.volume/n**3)
            comparisons.append(dict(case=name, volume_relative_l1_vs_inherited=relative_l1(r['volume'], inherited),
                                    max_relative_node_change=float(np.max(abs(r['volume']/inherited-1)))))
            np.savez_compressed(out/(name+'_comparison.npz'), inherited_volume=inherited, gaussian_volume=r['volume'])

    # Full-kernel generic integration, independently refined quadrature.
    for d in (1, 2, 3):
        scale = {1:1e-4, 2:0.01, 3:1.0}[d]
        b = Box((0,)*d, (0.1,)*d, measure_scale=scale)
        nodes = regular_nodes(b, 4, 0.65)
        nodes.centers += rng.uniform(-0.18, 0.18, nodes.centers.shape)*0.025
        if d == 2:
            angle = 0.63
            rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            covariance = rot @ np.diag([0.009**2, 0.021**2]) @ rot.T
            nodes = Nodes(nodes.centers, np.broadcast_to(covariance, nodes.covariance.shape), nodes.labels)
        ref = run(f'perturbed_{d}d_reference', nodes, [b], 24, order=4)
        errors = []
        for q in (8, 16, 32):
            r = run(f'perturbed_{d}d_q{q}', nodes, [b], q)
            error = relative_l1(r['volume'], ref['volume']); errors.append(error)
            convergence.append(dict(case=f'perturbed_{d}d', cells=q, order=1, volume_relative_l1=error))
        assert errors[-1] < errors[0]/8
        check_ref = run(f'perturbed_{d}d_reference_check', nodes, [b], 32, order=4)
        ref_error = relative_l1(ref['volume'], check_ref['volume'])
        assert ref_error < errors[-1]/10
        convergence.append(dict(case=f'perturbed_{d}d_reference_stability', cells=32, order=4, volume_relative_l1=ref_error))

    # Sparse/dense centers, explicitly independent from integration cells.
    b = Box((0,), (0.1,), measure_scale=1e-4)
    centers = np.array([0.005, 0.012, 0.019, 0.026, 0.04, 0.065, 0.09])[:, None]
    widths = np.array([0.007]*4+[0.012, 0.02, 0.02])
    nodes = Nodes(centers, widths[:, None, None]**2, np.zeros(7, int))
    nonuniform = run('nonuniform_nodes', nodes, [b], 512)
    fine = run('nonuniform_nodes_reference', nodes, [b], 512, order=4)
    assert relative_l1(nonuniform['volume'], fine['volume']) < 1e-4

    # Exact alignment of quadrature subdomains with a discontinuous phi interface.
    pieces = [Box((0,), (0.037,), measure_scale=1e-4), Box((0.037,), (0.1,), measure_scale=1e-4)]
    phi = lambda x, _: np.where(x[:, 0] < 0.037, 0.2, 0.8)
    expected = 1000*1e-4*(0.037*0.2+0.063*0.8)
    material = run('heterogeneous_phi', nodes, pieces, 128, phi=phi, order=3, analytic_capacity=expected)
    material_ref = run('heterogeneous_phi_reference', nodes, pieces, 256, phi=phi, order=3, analytic_capacity=expected)
    assert relative_l1(material['capacity'], material_ref['capacity']) < 1e-8
    naive_capacity = 1000*material['volume']*phi(nodes.centers, 0)
    report['center_sample_phi_capacity_relative_error'] = float(abs(naive_capacity.sum()-expected)/expected)
    np.savez_compressed(out/'heterogeneous_capacity_comparison.npz', integrated_capacity=material['capacity'], center_sample_capacity=naive_capacity)
    for count in (4, 8, 16):
        for sigma in (0.3, 1.2):
            alternative = regular_nodes(b, count, sigma)
            run(f'heterogeneous_n{count}_sigma{sigma}', alternative, pieces, 128,
                phi=phi, order=3, analytic_capacity=expected)
    zero = run('zero_porosity', nodes, [b], 128, phi=0)
    assert np.all(zero['capacity'] == 0)

    # Nearby disconnected domains, deliberately broad overlapping Gaussian tails.
    domains = [Box((0,), (0.1,), 0, 1e-4), Box((0.101,), (0.201,), 1, 1e-4)]
    parts = [regular_nodes(dom, 4, 5) for dom in domains]
    nodes = Nodes(np.concatenate([n.centers for n in parts]), np.concatenate([n.covariance for n in parts]), np.concatenate([n.labels for n in parts]))
    result = run('disconnected_domains', nodes, domains, 128, phi=lambda x, label: 0.2 if label == 0 else 0.8)
    report['domains'] = []
    for label, ph in [(0, 0.2), (1, 0.8)]:
        ids = nodes.labels == label
        assert abs(result['volume'][ids].sum()/domains[label].volume-1) < 1e-12
        assert abs(result['capacity'][ids].sum()/(1000*ph*domains[label].volume)-1) < 1e-12
        report['domains'].append(dict(label=label, volume=float(result['volume'][ids].sum()), capacity=float(result['capacity'][ids].sum())))

    # Edge/corner queries and far-field numerical underflow diagnostic.
    nodes = regular_nodes(box, 4)
    corners = np.array(list(__import__('itertools').product(*zip(box.lower, box.upper))))
    _, weights, distance = nodes.weights(corners, 0)
    assert np.max(abs(weights.sum(axis=1)-1)) < 1e-12
    _, farw, fard = nodes.weights(np.array([[100., 100., 100.]]), 0)
    report['coverage'] = dict(corner_max_nearest_mahalanobis=float(np.sqrt(distance.max())),
                              far_weights_sum=float(farw.sum()), far_nearest_mahalanobis=float(np.sqrt(fard[0])),
                              note='Far points normalize but fail the practical 3-sigma coverage diagnostic; no claim of good coverage.')

    # Original 6912-node scale. Cartesian equal-width kernels factorize exactly:
    # psi_ijk(x,y,z)=psi_i(x)psi_j(y)psi_k(z). This is full Gaussian quadrature,
    # not inherited cell volumes. Generic integration cross-checks the identity.
    shape = (24, 24, 12)
    large_nodes = regular_nodes(box, shape, 0.6)
    large = run('large_6912_generic', large_nodes, [box], 24, batch=64)
    tensor_volumes = []
    for q in (24, 48, 96, 192):
        tick = time.perf_counter(); factors = []
        for axis, n in enumerate(shape):
            one = Box((box.lower[axis],), (box.upper[axis],))
            one_nodes = regular_nodes(one, n, 0.6)
            factors.append(integrate(one_nodes, [one], q, order=4)['volume'])
        v = np.einsum('i,j,k->ijk', *factors).ravel()
        assert np.all(v > 0) and abs(v.sum()/box.volume-1) < 1e-10
        tensor_volumes.append(v)
        np.savez_compressed(out/f'large_6912_tensor_q{q}.npz', volume=v, capacity=600*v)
        convergence.append(dict(case='large_6912_tensor_GL4', cells=q, order=4,
                                volume_relative_l1=None, seconds=time.perf_counter()-tick))
    # Same midpoint rule as the generic computation, independent factorization.
    factors = [integrate(regular_nodes(Box((0,), (box.upper[a],)), n, 0.6),
                         [Box((0,), (box.upper[a],))], 24)['volume'] for a,n in enumerate(shape)]
    same_rule = np.einsum('i,j,k->ijk', *factors).ravel()
    factor_error = relative_l1(large['volume'], same_rule)
    assert factor_error < 1e-12
    for row, v in zip(convergence[-4:], tensor_volumes):
        row['volume_relative_l1'] = relative_l1(v, tensor_volumes[-1])
    assert relative_l1(tensor_volumes[-2], tensor_volumes[-1]) < 1e-8
    report['large_scale'] = dict(nodes=6912, shape=shape, factorization_relative_error=factor_error,
                                generic_midpoint_relative_l1_vs_reference=relative_l1(large['volume'], tensor_volumes[-1]),
                                reference_check_relative_l1=relative_l1(tensor_volumes[-2], tensor_volumes[-1]),
                                limitation='Tensor acceleration is exact only for this Cartesian product / separable-kernel / uniform-material case.')

    for filename, data in [('metrics.csv', rows), ('inherited_comparison.csv', comparisons), ('convergence.csv', convergence)]:
        keys = list(dict.fromkeys(k for row in data for k in row))
        with (out/filename).open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader(); writer.writerows(data)
    report.update(status='PASS', elapsed_seconds=time.perf_counter()-started, cases=rows,
                  inherited_comparison=comparisons, convergence=convergence)
    (out/'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print('PASS:', out.resolve(), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        if RUN_OUTPUT is not None:
            trace = traceback.format_exc()
            failure = RUN_OUTPUT / ('failure_'+time.strftime('%Y%m%d_%H%M%S')+'.json')
            failure.write_text(json.dumps(dict(status='FAIL', traceback=trace), indent=2), encoding='utf-8')
            report_path = RUN_OUTPUT/'report.json'
            current = json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else {}
            current.update(status='FAIL', failure_log=failure.name)
            report_path.write_text(json.dumps(current, indent=2, ensure_ascii=False), encoding='utf-8')
        raise
