import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'stage_m2_partition'))
from partition import Nodes,regular_nodes,Box
from gaussian_transport.geometry import box_domain,l_shape_domain,voronoi_2d,voronoi_box_3d,disjoint_union
from gaussian_transport.smooth_faces import smooth_geometry_2d,finite_difference_gradient_check
from gaussian_transport.transport import Transport,Material,exchange,linear_patch_flux
from gaussian_transport.remap import partition_volume,conservative_partition_remap


class PipelineTests(unittest.TestCase):
    def test_voronoi_box_geometry(self):
        centers=np.array([[.25,.25],[.75,.25],[.25,.75],[.75,.75]])
        g=voronoi_2d(centers,box_domain())
        np.testing.assert_allclose(g.volume,.25,atol=1e-13)
        self.assertAlmostEqual(g.face_area.sum(),2.)
        self.assertAlmostEqual(g.boundary_area.sum(),4.)
        self.assertEqual(len(g.edges),4)

    def test_l_domain_and_connectivity(self):
        domain=l_shape_domain(); centers=np.array([[.2,.2],[.7,.2],[.2,.7],[.35,.35]])
        g=voronoi_2d(centers,domain)
        self.assertAlmostEqual(g.volume.sum(),domain.volume,places=12)
        self.assertAlmostEqual(g.boundary_area.sum(),4.,places=12)
        self.assertTrue(len(g.edges)>=3)

    def test_disjoint_union_has_no_cross_edge(self):
        a=voronoi_2d(np.array([[.25,.5],[.75,.5]]),box_domain())
        b=voronoi_2d(np.array([[.25,.5],[.75,.5]]),box_domain())
        b.centers[:,0]+=1.01; b.centroid[:,0]+=1.01; b.face_center[:,0]+=1.01; b.boundary_center[:,0]+=1.01
        g=disjoint_union(a,b); self.assertFalse(np.any((g.edges[:,0]<2)&(g.edges[:,1]>=2)))

    def test_smooth_gradient_and_geometry(self):
        domain=box_domain(); b=Box((0,0),(1,1)); nodes=regular_nodes(b,(3,3),.7)
        err=finite_difference_gradient_check(nodes,np.array([[.23,.31],[.71,.62]]))
        self.assertLess(err,2e-8)
        g,raw=smooth_geometry_2d(nodes,domain,cells=36,order=4)
        self.assertLess(raw['metrics']['antisymmetry_max'],1e-14)
        self.assertLess(raw['metrics']['volume_relative_error'],1e-12)
        self.assertLess(raw['metrics']['closure_relative'],2e-8)
        self.assertGreater(len(g.edges),0)

    def test_limiter_and_closed_mass(self):
        mass=np.array([.8,.1,.9,0.]); cap=np.ones(4); i=np.array([0,0,1,2]); j=np.array([1,2,3,3])
        out,_=exchange(mass,cap,i,j,np.array([10.,4.,8.,6.]))
        self.assertAlmostEqual(out.sum(),mass.sum(),places=14); self.assertTrue(np.all((out>=0)&(out<=cap)))
        x=np.column_stack(((np.arange(8)+.5)/8,np.full(8,.05))); g=voronoi_2d(x,box_domain(height=.1,thickness=.1))
        tr=Transport(g,porosity=.6,material=Material(law='linear'),gravity=(0,0))
        state=tr.initial(.5+.2*np.cos(np.pi*x[:,0])); final,h,stats=tr.run(state,.05,max_dt=.002,tolerance=1e-7)
        self.assertLess(stats['max_mass_residual_kg'],1e-12); self.assertTrue(np.all((tr.saturation(final)>=0)&(tr.saturation(final)<=1)))

    def test_boundary_inventory_and_impermeable(self):
        x=np.array([[.25,.5],[.75,.5]]); g=voronoi_2d(x,box_domain())
        select=lambda p,n:n[:,1]>0.9
        tr=Transport(g,material=Material(permeability=0),gravity=(0,-9.81),boundary_selector=select,feed_rate=1e-5)
        final,_,stats=tr.run(tr.initial(free=1e-4),.5,max_dt=.05)
        self.assertEqual(final.pore.sum(),0); self.assertGreater(final.surface.sum(),0)
        self.assertLess(stats['max_mass_residual_kg'],1e-12)

    def test_patch_isotropic_and_anisotropic_failure(self):
        rng=np.random.default_rng(7); x=rng.uniform(.12,.88,(16,2)); g=voronoi_2d(x,box_domain())
        _,_,iso=linear_patch_flux(g,[.7,-.3],1.)
        K=np.array([[10.,3.],[3.,1.5]])
        _,_,bad=linear_patch_flux(g,[.7,-.3],permeability=K,method='tpfa')
        _,_,good=linear_patch_flux(g,[.7,-.3],permeability=K,method='reconstructed')
        self.assertLess(iso,1e-12); self.assertGreater(bad,1e-2); self.assertLess(good,1e-12)

    def test_constant_potential_has_zero_internal_flux(self):
        rng=np.random.default_rng(11); g=voronoi_2d(rng.uniform(.1,.9,(12,2)),box_domain())
        tr=Transport(g,material=Material(law='linear'),gravity=(0,0))
        _,_,rate=tr.rates(tr.initial(.4))
        np.testing.assert_allclose(rate[:len(g.edges)],0.,atol=1e-14)

    def test_3d_voronoi_volume(self):
        grid=np.array(np.meshgrid([.25,.75],[.25,.75],[.25,.75],indexing='ij')).reshape(3,-1).T
        g=voronoi_box_3d(grid)
        self.assertAlmostEqual(g.volume.sum(),1.,places=10); np.testing.assert_allclose(g.volume,.125,atol=1e-9)

    def test_global_split_merge_remap(self):
        domain=box_domain(); old=regular_nodes(Box((0,0),(1,1)),(3,3),.65)
        centers=np.vstack((old.centers,[[.57,.54]])); cov=np.concatenate((old.covariance,np.eye(2)[None]*.08**2))
        new=Nodes(centers,cov,np.zeros(len(centers),int)); vold=partition_volume(old,domain)
        mass=1000*.6*vold*(.2+.5*old.centers[:,0])
        mapped,cap,T,m=conservative_partition_remap(old,new,domain,mass)
        self.assertLess(m['relative_mass_error'],1e-12); self.assertLess(m['max_column_partition_error'],1e-12)
        self.assertLess(m['max_capacity_excess_kg'],1e-10)


if __name__=='__main__': unittest.main(verbosity=2)
