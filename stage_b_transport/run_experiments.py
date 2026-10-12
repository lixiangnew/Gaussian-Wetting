"""运行 B1–B4：数值验证、边界实验和候选研究方向初测。"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from geometry import box_model, boundary_face, subset
from solver import Material, State, Transport
from research import conservative_remap, query


def csv_write(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def line_svg(path, title, series, xlabel='time (s)', ylabel='mass (g)'):
    """简单静态折线图，不依赖 matplotlib；series=[(name,x,y),...]。"""
    colors = ['#1769aa', '#da7c16', '#31834a', '#9c3ca1']
    xmax = max(float(np.max(x)) for _, x, _ in series) or 1
    ymax = max(float(np.max(y)) for _, _, y in series) or 1
    ymax *= 1.08
    lines = ['<svg xmlns="http://www.w3.org/2000/svg" width="850" height="500">',
             '<rect width="850" height="500" fill="white"/>',
             f'<text x="70" y="30" font-family="sans-serif" font-size="18">{title}</text>',
             '<path d="M70 65 V420 H790" fill="none" stroke="black"/>']
    for tick in range(6):
        x = 70+720*tick/5; y = 420-350*tick/5
        lines.append(f'<text x="{x-10}" y="443" font-size="12">{xmax*tick/5:.3g}</text>')
        lines.append(f'<text x="8" y="{y+4}" font-size="12">{ymax*tick/5:.3g}</text>')
    for k, (name, xs, ys) in enumerate(series):
        pairs = list(zip(xs, ys)); stride = max(1, len(pairs)//700)
        shown = pairs[::stride] + [pairs[-1]]
        points = ' '.join(f'{70+720*x/xmax:.2f},{420-350*y/ymax:.2f}' for x,y in shown)
        color = colors[k % len(colors)]
        lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
        lines.append(f'<text x="{90+175*k}" y="60" fill="{color}" font-size="13">{name}</text>')
    lines.append(f'<text x="370" y="480">{xlabel}</text><text x="10" y="55">{ylabel}</text></svg>')
    path.write_text('\n'.join(lines), encoding='utf-8')


def analytic_experiment(out):
    rows = []
    length, end = 0.1, 120.0
    material = Material(law='linear', pressure_scale=1000, permeability=1e-12)
    diffusion = 1e-12*1000/(1e-3*0.6)
    for n in (16, 32, 64):
        model = box_model((n, 1, 1), (length, 0.01, 0.01))
        average_cos = np.cos(np.pi*model.centers[:, 0]/length)*np.sinc(1/(2*n))
        initial = 0.5+0.2*average_cos
        exact = 0.5+0.2*np.exp(-diffusion*(np.pi/length)**2*end)*average_cos
        transport = Transport(model, material, gravity=(0,0,0))
        state, history, stats = transport.run(transport.initial(initial), end,
                                             max_dt=0.04*(length/n)**2/diffusion, tolerance=1e-7)
        observed = transport.saturation(state)
        error = float(np.sqrt(np.average((observed-exact)**2, weights=model.volume)))
        rows.append({'n': n, 'l2_error': error, 'order': None if not rows else float(np.log2(rows[-1]['l2_error']/error)), **stats})
        np.savez_compressed(out/f'analytic_{n}.npz', x=model.centers[:,0], numerical=observed, exact=exact)
        print('E1 analytic', rows[-1], flush=True)
    assert rows[-1]['l2_error'] < rows[0]['l2_error']/4
    csv_write(out/'analytic_convergence.csv', rows)
    return rows


def wetting_case(out, name, bottom=False, no_gravity=False, impermeable=False,
                 surface_only=False, max_dt=0.3, eps=0.02, tolerance=1e-4,
                 initial_s=0.05, end_time=30):
    model = box_model()
    mat = Material(permeability=0 if impermeable else 2e-12, pressure_scale=800,
                   saturation_floor=eps)
    transport = Transport(model, mat, gravity=(0,0,0) if no_gravity else (0,0,-9.81),
                          face=boundary_face(model, side=-1 if bottom else 1), feed_rate=3e-5)
    if surface_only:
        transport.conductance[:] = 0  # Explicit ablation: boundary uptake, no internal transfer.
    start = transport.initial(initial_s, free=0.0012)
    initial_pore = start.pore.sum()
    state, history, stats = transport.run(start, end_time, max_dt=max_dt, tolerance=tolerance)
    csv_write(out/f'{name}_history.csv', history)
    np.savez_compressed(out/f'{name}_state.npz', centers=model.centers, saturation=transport.saturation(state),
                        pore_mass=state.pore, surface_mass=state.surface, free_mass=state.free,
                        capacity=model.capacity_mass)
    result = {'initial_s': initial_s, 'end_time_s': end_time, 'tolerance': tolerance,
              'eps': eps, 'max_dt_s': max_dt,
              'absorbed_kg': float(state.pore.sum()-initial_pore),
              'surface_kg': float(state.surface.sum()), 'free_kg': state.free,
              'max_s': float(transport.saturation(state).max()),
              'max_h_m': history[-1]['max_h_m'], **stats}
    assert stats['max_mass_residual_kg'] < 1e-12
    assert history[-1]['min_s'] >= -1e-10 and history[-1]['max_s'] <= 1+1e-10
    if impermeable:
        assert abs(result['absorbed_kg']) < 1e-15
    time = np.array([r['time_s'] for r in history])
    line_svg(out/f'{name}_inventory.svg', name,
             [(key, time, np.array([r[key] for r in history])*1000)
              for key in ('free_kg','surface_kg','pore_kg')])
    print('E2/E3', name, result, flush=True)
    return result, transport.saturation(state)


def geometry_experiments(out):
    base = box_model((20,1,1), (0.1,0.01,0.01))
    model = subset(base, np.abs(base.centers[:,0]-0.05) > 0.006)
    initial = np.where(model.centers[:,0] < 0.05, 0.8, 0.2)
    tr = Transport(model, Material(law='linear'), gravity=(0,0,0))
    state, _, stats = tr.run(tr.initial(initial), 10, max_dt=0.2)
    gap_error = float(np.max(np.abs(tr.saturation(state)-initial)))
    assert gap_error < 1e-12 and model.summary()['components'] == 2
    model = box_model((25,1,25), (0.1,0.004,0.1))
    x, z = model.centers[:,0]-0.05, model.centers[:,2]-0.05
    initial = 0.05+0.6*np.exp(-(x*x+z*z)/(2*0.004**2))
    tr = Transport(model, Material(law='linear', pressure_scale=1000,
                                  permeability=(1e-11,1e-12,1e-12)), gravity=(0,0,0))
    state, _, anis_stats = tr.run(tr.initial(initial), 0.5, max_dt=0.02, tolerance=1e-6)
    excess = (tr.saturation(state)-0.05)*model.volume
    variance_x = float(np.sum(excess*x*x)/excess.sum())
    variance_z = float(np.sum(excess*z*z)/excess.sum())
    assert variance_x > variance_z
    np.savez_compressed(out/'anisotropy.npz', centers=model.centers, saturation=tr.saturation(state))
    return {'air_gap_max_change': gap_error, 'air_gap_components': 2,
            'variance_x_m2': variance_x, 'variance_z_m2': variance_z,
            'variance_ratio': variance_x/variance_z, 'stats': anis_stats}


def remap_experiment(out):
    old = box_model((8,1,1), (0.1,0.01,0.01))
    new = box_model((13,1,1), (0.1,0.01,0.01))
    old.porosity[:] = np.where(old.centers[:,0] < 0.037, 0.25, 0.7)
    s = np.where(old.centers[:,0] < 0.037, 0.85, 0.12)
    old_mass = old.capacity_mass*s
    phi, mass, saturation = conservative_remap(old, new, old_mass)
    nearest = np.argmin(abs(new.centers[:,0,None]-old.centers[None,:,0]), axis=1)
    naive_mass = new.water_density*new.volume*old.porosity[nearest]*s[nearest]
    remap_error = float(abs(mass.sum()-old_mass.sum())/old_mass.sum())
    naive_error = float(abs(naive_mass.sum()-old_mass.sum())/old_mass.sum())
    assert remap_error < 1e-12
    assert (saturation >= 0).all() and (saturation <= 1+1e-12).all()
    assert abs(np.sum(new.volume*phi)-old.pore_volume.sum()) < 1e-16
    np.savez_compressed(out/'remap.npz', centers=new.centers, porosity=phi, saturation=saturation,
                        water_mass=mass, naive_mass=naive_mass)
    # Continue a uniform-porosity smooth problem after 16→32 remapping.
    a = box_model((16,1,1), (0.1,0.01,0.01)); b = box_model((32,1,1), (0.1,0.01,0.01))
    material = Material(law='linear', pressure_scale=1000)
    ta, tb = Transport(a,material,(0,0,0)), Transport(b,material,(0,0,0))
    initial_a = 0.5+0.2*np.cos(np.pi*a.centers[:,0]/0.1)*np.sinc(1/32)
    initial_b = 0.5+0.2*np.cos(np.pi*b.centers[:,0]/0.1)*np.sinc(1/64)
    mid, _, _ = ta.run(ta.initial(initial_a), 30, max_dt=0.1, tolerance=1e-7)
    _, transferred, _ = conservative_remap(a,b,mid.pore)
    final, _, _ = tb.run(State(transferred,np.empty(0),0),30,max_dt=0.1,tolerance=1e-7)
    reference, _, _ = tb.run(tb.initial(initial_b),60,max_dt=0.1,tolerance=1e-7)
    continuation_error = float(np.sqrt(np.mean((tb.saturation(final)-tb.saturation(reference))**2)))
    return {'conservative_relative_mass_error': remap_error,
            'nearest_relative_mass_error': naive_error,
            'continued_l2_vs_fine_run': continuation_error,
            'interpretation': 'Conservation is verified; remapping changes approximation error. Not a novelty claim.'}


def query_experiment(out):
    centers = np.array([[0.,0,0],[1.,0,0]])
    cov = np.repeat(np.eye(3)[None,:,:],2,axis=0)
    volume = np.ones(2); phi = np.array([0.2,0.8]); s = np.array([0.8,0.1])
    labels = np.zeros(2,dtype=int); points = np.array([[0.5,0,0]])
    def evaluate(c,v,ph,ss,co,la,weighted=True,direct=False):
        return float(query(c,co,v,ph,ss,la,points,np.array([0]),weighted,direct)[0,2])
    original = evaluate(centers,volume,phi,s,cov,labels)
    ids = np.array([0,0,0,0,1]); split_volume = np.array([0.25]*4+[1.])
    weighted = evaluate(centers[ids],split_volume,phi[ids],s[ids],cov[ids],labels[ids])
    naive = evaluate(centers[ids],split_volume,phi[ids],s[ids],cov[ids],labels[ids],False)
    direct = evaluate(centers,volume,phi,s,cov,labels,True,True)
    gated = evaluate(centers,volume,phi,s,cov,np.array([0,1]))
    assert abs(weighted-original) < 1e-12
    assert abs(naive-original) > 0.01
    assert abs(gated-0.8) < 1e-12
    refinement = []
    qx = np.linspace(0.02,0.08,61)
    qp = np.column_stack((qx,np.full_like(qx,0.005),np.full_like(qx,0.005)))
    exact = 0.5+0.2*np.cos(np.pi*qx/0.1)
    for n in (8,16,32):
        m = box_model((n,1,1),(0.1,0.01,0.01))
        values = 0.5+0.2*np.cos(np.pi*m.centers[:,0]/0.1)*np.sinc(1/(2*n))
        for mode in ('cell_scaled','fixed_width'):
            covariance = m.covariance.copy()
            if mode == 'fixed_width':
                covariance[:,0,0] = 0.015**2
            result = query(m.centers,covariance,m.volume,m.porosity,values,
                           np.zeros(n,dtype=int),qp,np.zeros(len(qp),dtype=int))[:,2]
            refinement.append({'n':n,'kernel':mode,'l2_error':float(np.sqrt(np.mean((result-exact)**2)))})
    csv_write(out/'query_refinement.csv',refinement)
    return {'original_s':original,'volume_split_s':weighted,'count_weighted_split_s':naive,
            'direct_s_interpolation':direct,'domain_gated_s':gated,'refinement':refinement,
            'interpretation':'Duplicate invariance is algebraic; real refinement and fixed-bandwidth bias are separate tests.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path(__file__).parent/'outputs')
    args = p.parse_args(); out = args.output; out.mkdir(parents=True,exist_ok=True)
    report = {'status':'running','parameters':'Synthetic SI parameters; see PLAN.md and source.'}
    report['analytic'] = analytic_experiment(out)
    wet = {}
    wet['top'], base_s = wetting_case(out,'top')
    wet['impermeable'], _ = wetting_case(out,'impermeable',impermeable=True)
    wet['surface_only'], _ = wetting_case(out,'surface_only',surface_only=True)
    wet['bottom_gravity'], _ = wetting_case(out,'bottom_gravity',bottom=True)
    wet['bottom_no_gravity'], _ = wetting_case(out,'bottom_no_gravity',bottom=True,no_gravity=True)
    wet['half_dt'], refined_s = wetting_case(out,'half_dt',max_dt=0.15)
    wet['epsilon_half'], _ = wetting_case(out,'epsilon_half',eps=0.01)
    wet['time_comparison'] = {'s_l2':float(np.sqrt(np.mean((base_s-refined_s)**2))),
                             'relative_absorption_difference':abs(wet['top']['absorbed_kg']-wet['half_dt']['absorbed_kg'])/max(wet['half_dt']['absorbed_kg'],1e-15)}
    report['wetting'] = wet
    report['geometry'] = geometry_experiments(out)
    report['remap'] = remap_experiment(out)
    report['query'] = query_experiment(out)
    report['status'] = 'completed'
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print('All experiment groups completed. Report:',(out/'report.json').resolve(),flush=True)


if __name__ == '__main__':
    main()
