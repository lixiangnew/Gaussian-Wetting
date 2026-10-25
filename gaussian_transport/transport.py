"""M4 conservative bounded graph transport for route-A/B geometries."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Material:
    permeability: object = 1e-12
    viscosity: float = 1e-3
    law: str = 'linear'
    pressure_scale: float = 1000.0
    vg_n: float = 2.0
    saturation_floor: float = 0.02

    def response(self,s):
        s=np.asarray(s,float); se=np.clip(s,0,1)
        if self.law=='linear': return self.pressure_scale*(1-se),np.ones_like(se)
        if self.law!='vg': raise ValueError('law must be linear or vg')
        m=1-1/self.vg_n; ev=np.maximum(se,self.saturation_floor)
        pc=self.pressure_scale*np.maximum(ev**(-1/m)-1,0)**(1/self.vg_n)
        kr=np.sqrt(se)*(1-(1-se**(1/m))**m)**2
        return pc,kr


@dataclass
class State:
    pore: np.ndarray
    surface: np.ndarray
    free: float
    def vector(self): return np.concatenate((self.pore,self.surface,[self.free]))


def exchange(mass,capacity,first,second,requested):
    donor=np.where(requested>=0,first,second); receiver=np.where(requested>=0,second,first); amount=abs(requested)
    outgoing=np.bincount(donor,weights=amount,minlength=len(mass)); incoming=np.bincount(receiver,weights=amount,minlength=len(mass))
    ro=np.ones_like(mass); ri=np.ones_like(mass)
    np.divide(np.maximum(mass,0),outgoing,out=ro,where=outgoing>0)
    np.divide(np.maximum(capacity-mass,0),incoming,out=ri,where=incoming>0)
    factor=np.minimum(1,np.minimum(ro[donor],ri[receiver])); moved=amount*factor*(1-1e-14)
    delta=np.zeros_like(mass); np.add.at(delta,donor,-moved); np.add.at(delta,receiver,moved)
    return mass+delta,int(np.count_nonzero((factor<1-1e-10)&(amount>0)))


class Transport:
    def __init__(self,geometry,porosity=.6,density=1000.,material=Material(),gravity=None,
                 boundary_selector=None,feed_rate=0.):
        self.g=geometry; self.n=len(geometry.centers); self.d=geometry.centers.shape[1]
        self.rho=float(density); self.phi=np.broadcast_to(np.asarray(porosity,float),(self.n,)).copy()
        if np.any((self.phi<0)|(self.phi>1)): raise ValueError('porosity outside [0,1]')
        self.capacity=self.rho*geometry.volume*self.phi; self.material=material
        self.gravity=np.zeros(self.d) if gravity is None else np.asarray(gravity,float)
        if self.gravity.shape!=(self.d,): raise ValueError('gravity dimension mismatch')
        k=np.asarray(material.permeability,float)
        if k.ndim==0: k=np.broadcast_to(np.eye(self.d)*k,(self.n,self.d,self.d)).copy()
        elif k.shape==(self.d,self.d): k=np.broadcast_to(k,(self.n,self.d,self.d)).copy()
        elif k.shape!=(self.n,self.d,self.d): raise ValueError('invalid permeability')
        if not np.isfinite(k).all() or np.any(np.linalg.eigvalsh(k)<-1e-20): raise ValueError('permeability must be PSD')
        self.k=k; self.i,self.j=geometry.edges.T
        nvec=geometry.face_normal
        ki=np.einsum('ed,edk,ek->e',nvec,k[self.i],nvec); kj=np.einsum('ed,edk,ek->e',nvec,k[self.j],nvec)
        kh=np.divide(2*ki*kj,ki+kj,out=np.zeros_like(ki),where=ki+kj>0)
        self.conductance=self.rho/material.viscosity*geometry.face_area/geometry.center_distance*kh
        selected=np.arange(len(geometry.boundary_node))
        if boundary_selector is not None:
            selected=np.flatnonzero(boundary_selector(geometry.boundary_center,geometry.boundary_normal))
        self.bn=geometry.boundary_node[selected]; self.ba=geometry.boundary_area[selected]
        self.bp=geometry.boundary_center[selected]; self.bnormal=geometry.boundary_normal[selected]
        half=np.abs(np.einsum('bd,bd->b',geometry.centers[self.bn]-self.bp,self.bnormal))
        self.half=np.maximum(half,1e-15); kn=np.einsum('bd,bdk,bk->b',self.bnormal,k[self.bn],self.bnormal)
        self.boundary_conductance=self.rho/material.viscosity*self.ba/self.half*kn
        self.feed_rate=float(feed_rate)
        if self.feed_rate>0 and not len(self.bn): raise ValueError('feed requires selected boundary')

    def initial(self,saturation=0.,free=0.,surface=0.):
        s=np.broadcast_to(np.asarray(saturation,float),(self.n,)).copy(); surf=np.broadcast_to(np.asarray(surface,float),(len(self.bn),)).copy()
        if np.any((s<0)|(s>1)) or np.any(surf<0) or free<0: raise ValueError('invalid state')
        return State(s*self.capacity,surf,float(free))

    def saturation(self,state):
        return np.divide(state.pore,self.capacity,out=np.zeros(self.n),where=self.capacity>0)

    def rates(self,state):
        pc,kr=self.material.response(self.saturation(state)); H=-pc-self.rho*(self.g.centers@self.gravity)
        rate=self.conductance*.5*(kr[self.i]+kr[self.j])*(H[self.i]-H[self.j])
        first=self.i.copy(); second=self.j.copy()
        if len(self.bn):
            h=state.surface/(self.rho*self.ba)
            Hs=self.rho*np.linalg.norm(self.gravity)*h-self.rho*(self.bp@self.gravity)
            absorb=self.boundary_conductance*.5*(1+kr[self.bn])*np.maximum(Hs-H[self.bn],0)
            patch=self.n+np.arange(len(self.bn)); freeid=self.n+len(self.bn)
            first=np.r_[first,patch,np.full(len(self.bn),freeid)]; second=np.r_[second,self.bn,patch]
            feed=np.full(len(self.bn),self.feed_rate/len(self.bn))
            rate=np.r_[rate,absorb,feed]
        return first,second,rate

    def step(self,state,dt):
        first,second,rate=self.rates(state); capacity=np.r_[self.capacity,np.full(len(self.bn)+1,np.inf)]
        out,limited=exchange(state.vector(),capacity,first,second,rate*dt)
        tol=max(self.capacity.sum(),1e-12)*1e-11
        if not np.isfinite(out).all() or out.min()<-tol or np.max(out[:self.n]-self.capacity)>tol: raise FloatingPointError('invalid bounded update')
        return State(out[:self.n],out[self.n:self.n+len(self.bn)],float(out[-1])),limited

    def run(self,state,end_time,max_dt=.1,tolerance=1e-6,max_steps=500000):
        initial=float(state.vector().sum()); time=0.; dt=max_dt; accepted=rejected=limited=0; history=[]
        scale=np.r_[np.maximum(self.capacity,1e-18),np.maximum(self.capacity[self.bn],1e-12),max(initial,1e-12)]
        def record():
            s=self.saturation(state); history.append(dict(time_s=time,pore_kg=float(state.pore.sum()),surface_kg=float(state.surface.sum()),free_kg=state.free,
              mass_residual_kg=float(state.vector().sum()-initial),min_s=float(s.min()),max_s=float(s.max())))
        record()
        while time<end_time and accepted+rejected<max_steps:
            dt=min(dt,max_dt,end_time-time); full,_=self.step(state,dt); half,l1=self.step(state,dt/2); refined,l2=self.step(half,dt/2)
            error=float(np.max(abs(full.vector()-refined.vector())/scale))
            if error>tolerance:
                rejected+=1; dt*=max(.1,.8*np.sqrt(tolerance/error)); continue
            state=refined; time+=dt; accepted+=1; limited+=l1+l2; record(); dt*=min(2.,.9*np.sqrt(tolerance/max(error,1e-30)))
        if time<end_time: raise RuntimeError('max steps exceeded')
        return state,history,dict(accepted_steps=accepted,rejected_steps=rejected,limited_updates=limited,
          max_mass_residual_kg=max(abs(r['mass_residual_kg']) for r in history))


def linear_patch_flux(geometry,gradient,permeability=1.,density=1.,viscosity=1.,method='tpfa'):
    """Compare pair flux to exact linear-field flux; reconstructed is multipoint LS."""
    grad=np.asarray(gradient,float); H=geometry.centers@grad; i,j=geometry.edges.T; n=geometry.face_normal
    K=np.asarray(permeability,float)
    if K.ndim==0: K=np.eye(len(grad))*K
    exact=-density/viscosity*geometry.face_area*np.einsum('ed,dk,k->e',n,K,grad)
    if method=='tpfa':
        kn=np.einsum('ed,dk,ek->e',n,K,n)
        numerical=density/viscosity*geometry.face_area/geometry.center_distance*kn*(H[i]-H[j])
    elif method=='reconstructed':
        # Global linear least-squares is the exact first-order reconstruction used as an M6 diagnostic.
        design=np.c_[np.ones(len(H)),geometry.centers]; fitted=np.linalg.lstsq(design,H,rcond=None)[0][1:]
        numerical=-density/viscosity*geometry.face_area*np.einsum('ed,dk,k->e',n,K,fitted)
    else: raise ValueError('unknown method')
    scale=max(np.linalg.norm(exact),1e-30)
    return numerical,exact,float(np.linalg.norm(numerical-exact)/scale)
