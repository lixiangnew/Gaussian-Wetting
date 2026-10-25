"""M6 candidate: global conservative remap after Gaussian split/merge."""
import numpy as np
from .smooth_faces import _rectangle_quadrature


def partition_volume(nodes,domain,cells=64,order=4):
    v=np.zeros(len(nodes.centers))
    for piece in domain.pieces:
        p,q=_rectangle_quadrature(piece,cells,order); ids,w,_=nodes.weights(p,0)
        v[ids]+=(q*domain.thickness)@w
    return v


def conservative_partition_remap(old_nodes,new_nodes,domain,old_mass,porosity=.6,density=1000.,cells=64,order=4):
    old_mass=np.asarray(old_mass,float)
    if old_mass.shape!=(len(old_nodes.centers),) or np.any(old_mass<0): raise ValueError('invalid old mass')
    vold=np.zeros(len(old_mass)); vnew=np.zeros(len(new_nodes.centers)); joint=np.zeros((len(vnew),len(vold)))
    for piece in domain.pieces:
        p,q=_rectangle_quadrature(piece,cells,order); q=q*domain.thickness
        io,wo,_=old_nodes.weights(p,0); inn,wn,_=new_nodes.weights(p,0)
        vold[io]+=q@wo; vnew[inn]+=q@wn
        joint[np.ix_(inn,io)]+=np.einsum('q,qj,qi->ji',q,wn,wo,optimize=True)
    transfer=np.divide(joint,vold[None,:],out=np.zeros_like(joint),where=vold[None,:]>0)
    new_mass=transfer@old_mass; capacity=density*porosity*vnew
    metrics=dict(relative_mass_error=abs(new_mass.sum()-old_mass.sum())/max(old_mass.sum(),1e-30),
                 max_column_partition_error=float(np.max(abs(transfer.sum(axis=0)-1))),
                 max_capacity_excess_kg=float(np.max(new_mass-capacity)),
                 old_volume=float(vold.sum()),new_volume=float(vnew.sum()))
    return new_mass,capacity,transfer,metrics
