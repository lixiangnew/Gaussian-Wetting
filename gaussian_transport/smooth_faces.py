"""M3 route B: numerically integrated smooth-Gaussian effective faces.

The effective-area identity is literature-derived for partition-of-unity
meshless FV formulations. Its use as a two-point Darcy conductance here is a
project candidate and is accepted only when projection/patch diagnostics pass.
"""
from dataclasses import dataclass
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'stage_m2_partition'))
from partition import Nodes
from .geometry import GraphGeometry, _piece_constraints


def weights_gradients(nodes, points, label=0):
    points=np.asarray(points,float); ids,w,nearest=nodes.weights(points,label)
    invcov=np.linalg.inv(nodes.covariance[ids])
    delta=points[:,None,:]-nodes.centers[ids]
    gradlog=-np.einsum('nij,bnj->bni',invcov,delta,optimize=True)
    mean=np.einsum('bn,bnd->bd',w,gradlog,optimize=True)
    grad=w[:,:,None]*(gradlog-mean[:,None,:])
    return ids,w,grad,nearest


def _rectangle_quadrature(piece,cells,order):
    lo=piece.min(axis=0); hi=piece.max(axis=0)
    z,w=np.polynomial.legendre.leggauss(order); axes=[]; weights=[]
    for a,b in zip(lo,hi):
        h=(b-a)/cells
        axes.append((a+(np.arange(cells)[:,None]+.5)*h+.5*h*z).ravel())
        weights.append(np.tile(.5*h*w,cells))
    xx,yy=np.meshgrid(axes[0],axes[1],indexing='ij'); wx,wy=np.meshgrid(weights[0],weights[1],indexing='ij')
    return np.column_stack((xx.ravel(),yy.ravel())),(wx*wy).ravel()


def smooth_geometry_2d(nodes,domain,cells=24,order=3,projection_tol=1e-12):
    n=len(nodes.centers); d=2
    if nodes.centers.shape[1]!=2 or set(nodes.labels.tolist())!={0}: raise ValueError('single-label 2D nodes required')
    volume=np.zeros(n); first=np.zeros((n,d)); moment=np.zeros((n,n,d))
    for piece in domain.pieces:
        points,qw=_rectangle_quadrature(piece,cells,order)
        ids,psi,grad,_=weights_gradients(nodes,points,0)
        qw=qw*domain.thickness
        volume[ids]+=qw@psi
        first[ids]+=np.einsum('q,qn,qd->nd',qw,psi,points,optimize=True)
        moment[np.ix_(ids,ids,np.arange(d))]+=np.einsum('q,qi,qjd->ijd',qw,psi,grad,optimize=True)
    area=moment-moment.transpose(1,0,2)
    boundary=np.zeros((n,d)); b_rows=[]
    z,w=np.polynomial.legendre.leggauss(max(order,4))
    p=domain.boundary
    for a,b in zip(p,np.roll(p,-1,axis=0)):
        vec=b-a; length=np.linalg.norm(vec); tangent=vec/length; normal=np.array([tangent[1],-tangent[0]])
        fractions=((np.arange(cells)[:,None]+.5+.5*z[None,:])/cells).ravel()
        points=a[None,:]+fractions[:,None]*vec
        ids,psi,_,_=weights_gradients(nodes,points,0); qw=.5*length*w*domain.thickness
        qw=np.tile(qw/cells,cells)
        contribution=qw@psi
        boundary[ids]+=contribution[:,None]*normal
        for local,node in enumerate(ids):
            if contribution[local]>projection_tol:
                center=np.average(points,axis=0,weights=qw*psi[:,local])
                b_rows.append((node,contribution[local],normal,center))
    rows=[]; negative=0
    for i in range(n):
        for j in range(i+1,n):
            delta=nodes.centers[j]-nodes.centers[i]; dist=np.linalg.norm(delta); e=delta/dist
            projected=float(area[i,j]@e)
            if projected< -projection_tol: negative+=1
            if projected>projection_tol:
                rows.append((i,j,projected,e,.5*(nodes.centers[i]+nodes.centers[j]),dist))
    edges=np.array([[r[0],r[1]] for r in rows],int).reshape(-1,2)
    centroid=first/volume[:,None]
    g=GraphGeometry(nodes.centers.copy(),volume,centroid,edges,np.array([r[2] for r in rows]),
      np.array([r[3] for r in rows]).reshape(-1,2),np.array([r[4] for r in rows]).reshape(-1,2),
      np.array([r[5] for r in rows]),np.array([r[0] for r in b_rows],int),np.array([r[1] for r in b_rows]),
      np.array([r[2] for r in b_rows]).reshape(-1,2),np.array([r[3] for r in b_rows]).reshape(-1,2),
      'smooth_gaussian_projected_2d').validate()
    closure=area.sum(axis=1)+boundary
    metrics=dict(antisymmetry_max=float(np.max(abs(area+area.transpose(1,0,2)))),
                 closure_linf=float(np.max(np.linalg.norm(closure,axis=1))),
                 closure_relative=float(np.max(np.linalg.norm(closure,axis=1))/max(np.linalg.norm(boundary,axis=1).max(),1e-30)),
                 negative_projected_pairs=negative, retained_edges=len(rows),
                 volume_relative_error=abs(volume.sum()-domain.volume)/domain.volume)
    return g,dict(area_vector=area,boundary_vector=boundary,closure=closure,metrics=metrics)


def finite_difference_gradient_check(nodes,points,eps=1e-7):
    _,w,g,_=weights_gradients(nodes,points,0); numeric=np.zeros_like(g)
    for d in range(points.shape[1]):
        offset=np.zeros(points.shape[1]); offset[d]=eps
        _,wp,_=nodes.weights(points+offset,0); _,wm,_=nodes.weights(points-offset,0)
        numeric[:,:,d]=(wp-wm)/(2*eps)
    return float(np.max(abs(g-numeric)))
