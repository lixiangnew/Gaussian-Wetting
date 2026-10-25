"""Run M3--M6 validation and write raw JSON/CSV plus stage reports."""
import csv,json,platform,subprocess,time,tracemalloc,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'stage_m2_partition'))
from partition import Box,Nodes,regular_nodes,integrate
from gaussian_transport.geometry import box_domain,l_shape_domain,voronoi_2d,voronoi_box_3d,disjoint_union
from gaussian_transport.smooth_faces import smooth_geometry_2d,finite_difference_gradient_check
from gaussian_transport.transport import Transport,Material,linear_patch_flux
from gaussian_transport.remap import partition_volume,conservative_partition_remap

SEED=20261023


def write_csv(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)


def measured(fn):
    tracemalloc.start(); t=time.perf_counter(); value=fn(); seconds=time.perf_counter()-t; _,peak=tracemalloc.get_traced_memory(); tracemalloc.stop()
    return value,seconds,peak


def strip_geometry(n,perturb=0.,rng=None,L=.1,area=.0001):
    x=(np.arange(n)+.5)*L/n
    if perturb:
        x+=rng.uniform(-perturb,perturb,n)*L/n; x.sort()
    centers=np.column_stack((x,np.full(n,.005)))
    return voronoi_2d(centers,box_domain(L,.01,area/.01)),x


def cell_cos_average(x,L=.1):
    bounds=np.r_[0,.5*(x[:-1]+x[1:]),L]; width=np.diff(bounds)
    return (np.sin(np.pi*bounds[1:]/L)-np.sin(np.pi*bounds[:-1]/L))/(np.pi*width/L)


def fixed_run(tr,state,end,dt):
    initial=state.vector().sum(); steps=int(np.ceil(end/dt)); dt=end/steps
    for _ in range(steps): state,_=tr.step(state,dt)
    return state,steps,float(state.vector().sum()-initial)


def graph_checks(g):
    degree=np.bincount(g.edges.ravel(),minlength=len(g.centers)) if len(g.edges) else np.zeros(len(g.centers),int)
    return dict(nodes=len(g.centers),edges=len(g.edges),average_neighbors=float(degree.mean()),
      self_loops=int(np.count_nonzero(g.edges[:,0]==g.edges[:,1])),
      duplicate_edges=int(len(g.edges)-len(np.unique(g.edges,axis=0))))


def threshold_crossing(x,values,level):
    order=np.argsort(x); x=x[order]; y=np.asarray(values)[order]-level
    hit=np.flatnonzero(y[:-1]*y[1:]<=0)
    if not len(hit): return None
    k=hit[np.argmin(abs(.5*(x[hit]+x[hit+1])-.5*(x[0]+x[-1])))]
    if abs(y[k+1]-y[k])<1e-30: return float(.5*(x[k]+x[k+1]))
    return float(x[k]-y[k]*(x[k+1]-x[k])/(y[k+1]-y[k]))


def main():
    out=ROOT/'gaussian_transport'/'outputs'; reports=ROOT/'reports'; out.mkdir(exist_ok=True); reports.mkdir(exist_ok=True)
    rng=np.random.default_rng(SEED); commit=None
    try: commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
    except Exception: pass
    report=dict(status='RUNNING',seed=SEED,git_commit=commit,python=platform.python_version(),numpy=np.__version__,
      scope='M3-M6 independent CPU prototype; old external FVM parity remains BLOCKED')
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')

    # M3 route A: clipped Voronoi on regular, irregular, and concave domains.
    centers=np.array([[.25,.25],[.75,.25],[.25,.75],[.75,.75]])
    boxg=voronoi_2d(centers,box_domain()); lcent=np.array([[.15,.15],[.5,.18],[.82,.2],[.18,.5],[.2,.82],[.35,.35]])
    lg=voronoi_2d(lcent,l_shape_domain())
    route_a=dict(box_volume_error=abs(boxg.volume.sum()-1),box_boundary_error=abs(boxg.boundary_area.sum()-4),
      box_edges=len(boxg.edges),box_graph=graph_checks(boxg),l_volume_error=abs(lg.volume.sum()-.64),
      l_boundary_error=abs(lg.boundary_area.sum()-4),l_edges=len(lg.edges),l_graph=graph_checks(lg))

    # M3 route B: smooth Gaussian area vectors and candidate two-point failure case.
    b=Box((0,0),(1,1)); base=regular_nodes(b,(4,4),.7)
    smooth_regular,raw_regular=smooth_geometry_2d(base,box_domain(),32,3)
    smooth_reference,raw_reference=smooth_geometry_2d(base,box_domain(),48,4)
    rotated=[]
    for theta in rng.uniform(0,np.pi,len(base.centers)):
        R=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        rotated.append(R@np.diag([.12**2,.25**2])@R.T)
    anodes=Nodes(base.centers,np.array(rotated),base.labels)
    smooth_aniso,raw_aniso=smooth_geometry_2d(anodes,box_domain(),32,3)
    grad=np.array([.7,-.3]); av=raw_aniso['area_vector'][smooth_aniso.edges[:,0],smooth_aniso.edges[:,1]]
    candidate=-smooth_aniso.face_area*(smooth_aniso.face_normal@grad); exact=-av@grad
    candidate_error=float(np.linalg.norm(candidate-exact)/np.linalg.norm(exact))
    quadrature=dict(volume_relative_l2=float(np.linalg.norm(smooth_regular.volume-smooth_reference.volume)/np.linalg.norm(smooth_reference.volume)),
      area_vector_relative_l2=float(np.linalg.norm(raw_regular['area_vector']-raw_reference['area_vector'])/np.linalg.norm(raw_reference['area_vector'])),
      boundary_vector_relative_l2=float(np.linalg.norm(raw_regular['boundary_vector']-raw_reference['boundary_vector'])/np.linalg.norm(raw_reference['boundary_vector'])))
    avnorm=np.linalg.norm(raw_aniso['area_vector'],axis=2); maxav=float(avnorm.max()); pruning=[]
    for fraction in (0.,.01,.05,.1):
        keep=avnorm>=fraction*maxav; pruned=np.where(keep[:,:,None],raw_aniso['area_vector'],0.)
        residual=pruned.sum(axis=1)+raw_aniso['boundary_vector']; denom=max(np.linalg.norm(raw_aniso['boundary_vector'],axis=1).max(),1e-30)
        pruning.append(dict(relative_threshold=fraction,retained_pairs=int(np.count_nonzero(np.triu(keep,1))),
          closure_relative=float(np.linalg.norm(residual,axis=1).max()/denom)))
    route_b=dict(gradient_fd_error=finite_difference_gradient_check(anodes,np.array([[.21,.31],[.77,.64]])),
      regular=raw_regular['metrics'],anisotropic_kernel=raw_aniso['metrics'],
      quadrature_crosscheck=quadrature,boundary_perimeter_error=abs(smooth_reference.boundary_area.sum()-4.),
      pruning_ablation=pruning,
      anisotropic_kernel_projected_tpfa_patch_error=candidate_error,
      decision='FAIL for general two-point transport; retain geometry candidate, do not use in main solver')

    # M4/M5 analytical diffusion, route A. Same physical parameters for every resolution.
    convergence=[]; phi=.6; k=1e-12; p0=1000.; mu=1e-3; D=k*p0/(mu*phi); end=120.; mean=.5; amp=.2
    for layout,perturb in [('regular',0.),('perturbed',.22)]:
        for n in (16,32,64):
            def build(): return strip_geometry(n,perturb,rng)
            (g,x),prep,peak=measured(build)
            tr=Transport(g,phi,material=Material(k,mu,'linear',p0),gravity=(0,0))
            initial=mean+amp*cell_cos_average(x); dt=.15*(.1/n)**2/D
            solve_start=time.perf_counter(); state,steps,residual=fixed_run(tr,tr.initial(initial),end,dt); solve_time=time.perf_counter()-solve_start
            exact_s=mean+amp*np.exp(-D*(np.pi/.1)**2*end)*cell_cos_average(x)
            observed=tr.saturation(state); l1=float(np.average(abs(observed-exact_s),weights=g.volume)); l2=float(np.sqrt(np.average((observed-exact_s)**2,weights=g.volume)))
            crossing=threshold_crossing(x,observed,mean); front_error=None if crossing is None else abs(crossing-.05)
            convergence.append(dict(layout=layout,n=n,l1=l1,l2=l2,steps=steps,dt_s=end/steps,mass_residual_kg=residual,
              front_error_m=front_error,min_mass_kg=float(state.pore.min()),max_capacity_excess_kg=float(np.max(state.pore-tr.capacity)),
              preprocess_s=prep,preprocess_peak_bytes=peak,solve_s=solve_time,edges=len(g.edges),average_neighbors=graph_checks(g)['average_neighbors']))
    for layout in ('regular','perturbed'):
        vals=[r for r in convergence if r['layout']==layout]
        if not all(vals[i+1]['l2']<vals[i]['l2'] for i in range(2)): raise AssertionError('analytic convergence failed')
        if max(abs(r['mass_residual_kg']) for r in vals)>1e-12: raise AssertionError('mass ledger failed')

    # Independent time-step refinement at fixed n=32.
    g,x=strip_geometry(32,0,rng); tr=Transport(g,phi,material=Material(k,mu,'linear',p0),gravity=(0,0)); initial=mean+amp*cell_cos_average(x)
    time_rows=[]; temporal=[]
    for dt in (.2,.1,.05,.00625):
        state,steps,res=fixed_run(tr,tr.initial(initial),end,dt); temporal.append((dt,state,steps,res))
    temporal_reference=tr.saturation(temporal[-1][1])
    for dt,state,steps,res in temporal[:-1]:
        err=float(np.sqrt(np.mean((tr.saturation(state)-temporal_reference)**2)))
        time_rows.append(dict(dt_s=end/steps,l2_vs_dt_0_00625=err,steps=steps,mass_residual_kg=res))
    if not (time_rows[2]['l2_vs_dt_0_00625']<time_rows[1]['l2_vs_dt_0_00625']<time_rows[0]['l2_vs_dt_0_00625']): raise AssertionError('time refinement failed')

    # Boundary reservoirs: free -> surface -> pore, and impermeable retention.
    g,_=strip_geometry(12,0,rng); top=lambda p,n:n[:,1]>.9
    wet=Transport(g,phi,material=Material(2e-12,mu,'vg',800.),gravity=(0,-9.81),boundary_selector=top,feed_rate=3e-5)
    wet_final,wet_history,wet_stats=wet.run(wet.initial(.05,free=.0012),2.,max_dt=.05,tolerance=1e-4)
    dry=Transport(g,phi,material=Material(0.,mu,'linear',p0),gravity=(0,-9.81),boundary_selector=top,feed_rate=3e-5)
    dry_final,dry_history,dry_stats=dry.run(dry.initial(0,free=.0012),2.,max_dt=.05,tolerance=1e-5)
    ledger=dict(wet=dict(pore=float(wet_final.pore.sum()),surface=float(wet_final.surface.sum()),free=wet_final.free,**wet_stats),
      impermeable=dict(pore=float(dry_final.pore.sum()),surface=float(dry_final.surface.sum()),free=dry_final.free,**dry_stats))
    if dry_final.pore.sum()!=0 or max(wet_stats['max_mass_residual_kg'],dry_stats['max_mass_residual_kg'])>1e-12: raise AssertionError('boundary ledger failed')

    # Constant potential and isotropic/anisotropic linear patch tests.
    points=rng.uniform(.08,.92,(24,2)); vg=voronoi_2d(points,box_domain())
    _,_,iso=linear_patch_flux(vg,[.6,-.2],1.)
    K=np.array([[10.,3.],[3.,1.5]]); _,_,tpfa_bad=linear_patch_flux(vg,[.6,-.2],K,method='tpfa'); _,_,reconstructed=linear_patch_flux(vg,[.6,-.2],K,method='reconstructed')
    constant_solver=Transport(vg,phi,material=Material(k,mu,'linear',p0),gravity=(0,0))
    _,_,constant_rate=constant_solver.rates(constant_solver.initial(.5)); internal_rate=constant_rate[:len(vg.edges)]
    constant_max=float(np.max(abs(internal_rate)))
    if constant_max>1e-14: raise AssertionError('constant potential internal flux failed')
    patch=dict(constant_potential_max_rate=constant_max,isotropic_tpfa_relative_error=iso,anisotropic_tpfa_relative_error=tpfa_bad,
      anisotropic_reconstructed_relative_error=reconstructed,
      decision='Use TPFA only for scalar/K-orthogonal cases; use multipoint gradient reconstruction diagnostic for general anisotropy.')

    # Nearby disconnected components: explicit topology prevents cross-gap transport.
    left=voronoi_2d(np.array([[.2,.5],[.7,.5]]),box_domain())
    right0=voronoi_2d(np.array([[.2,.5],[.7,.5]]),box_domain())
    right0.centers[:,0]+=1.01; right0.centroid[:,0]+=1.01; right0.face_center[:,0]+=1.01; right0.boundary_center[:,0]+=1.01
    union=disjoint_union(left,right0); cut=len(left.centers)
    cross=int(np.count_nonzero((union.edges[:,0]<cut)&(union.edges[:,1]>=cut)))
    disconnected=dict(components=2,gap_m=.01,cross_component_edges=cross)
    if cross: raise AssertionError('disconnected components were joined')

    # M6 fixed irregular 3D comparison: explicit Voronoi volume vs Gaussian partition volume.
    grid=np.array(np.meshgrid([.18,.5,.82],[.18,.5,.82],[.18,.5,.82],indexing='ij')).reshape(3,-1).T
    grid+=rng.uniform(-.035,.035,grid.shape)
    (g3,pre3,peak3)=(*measured(lambda:voronoi_box_3d(grid)),) # value, seconds, bytes
    cov=np.broadcast_to(np.eye(3)*.18**2,(len(grid),3,3)).copy(); gn=Nodes(grid,cov,np.zeros(len(grid),int))
    part=integrate(gn,[Box((0,0,0),(1,1,1))],28,order=2,batch_size=256)
    volume_l1=float(np.sum(abs(g3.volume-part['volume']))/g3.volume.sum())
    three_d=dict(nodes=len(grid),voronoi_volume_error=abs(g3.volume.sum()-1),gaussian_partition_volume_error=abs(part['volume'].sum()-1),
      volume_definition_relative_l1=volume_l1,voronoi_preprocess_s=pre3,voronoi_peak_bytes=peak3,edges=len(g3.edges))

    # M6 split/merge global remap candidate.
    old=regular_nodes(Box((0,0),(1,1)),(4,4),.65); vold=partition_volume(old,box_domain()); oldmass=1000*.6*vold*(.15+.65*old.centers[:,0])
    centers=np.vstack((old.centers,[[.53,.52]])); cov=np.concatenate((old.covariance,np.eye(2)[None]*.09**2)); new=Nodes(centers,cov,np.zeros(len(centers),int))
    newmass,newcap,T,split=conservative_partition_remap(old,new,box_domain(),oldmass)
    back,backcap,T2,merge=conservative_partition_remap(new,old,box_domain(),newmass)
    adaptive=dict(split=split,merge=merge,roundtrip_saturation_l2=float(np.sqrt(np.mean((back/backcap-oldmass/(1000*.6*vold))**2))),
      classification='project candidate; conservative but introduces remap diffusion')

    report.update(status='PASS_WITH_REJECTED_CANDIDATES',route_a=route_a,route_b=route_b,convergence=convergence,
      time_convergence=time_rows,ledger=ledger,patch_tests=patch,disconnected=disconnected,three_d=three_d,adaptive_remap=adaptive,
      blockers=['Original external FVM source unavailable: M0/M1 strict parity BLOCKED.',
                'Smooth Gaussian projected two-point flux rejected for anisotropic kernels.',
                'Full MPFA nonlinear Richards solver and production 3D complex-geometry mesher remain future work.'])
    (out/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    write_csv(out/'spatial_convergence.csv',convergence); write_csv(out/'time_convergence.csv',time_rows)
    write_csv(out/'route_b_pruning.csv',pruning)
    write_csv(out/'wet_history.csv',wet_history); write_csv(out/'impermeable_history.csv',dry_history)
    print(json.dumps({k:report[k] for k in ('status','route_a','route_b','patch_tests','three_d','adaptive_remap')},indent=2,ensure_ascii=False))


if __name__=='__main__':
    try: main()
    except Exception as exc:
        out=ROOT/'gaussian_transport'/'outputs'; current={}
        if (out/'report.json').exists(): current=json.loads((out/'report.json').read_text(encoding='utf-8'))
        current.update(status='FAIL',error=repr(exc)); (out/'report.json').write_text(json.dumps(current,indent=2,ensure_ascii=False),encoding='utf-8')
        raise
