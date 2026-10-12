import unittest
import numpy as np
from geometry import box_model, boundary_face
from solver import Material, Transport, exchange
from research import conservative_remap


class TransportTests(unittest.TestCase):
    def test_limiter_mass_and_bounds(self):
        mass = np.array([0.8,0.1,0.9,0.0]); capacity = np.ones(4)
        i = np.array([0,0,1,2]); j = np.array([1,2,3,3]); requests = np.array([10.,4.,8.,6.])
        out,_ = exchange(mass,capacity,i,j,requests)
        self.assertAlmostEqual(out.sum(),mass.sum(),places=14)
        self.assertTrue((out >= 0).all() and (out <= capacity).all())
        order = np.array([3,1,0,2])
        reordered,_ = exchange(mass,capacity,i[order],j[order],requests[order])
        np.testing.assert_allclose(out,reordered,atol=1e-14)

    def test_naive_clipping_loses_mass(self):
        # A deliberately wrong baseline: clipping an oversized two-node transfer.
        mass = np.array([0.8,0.1]); proposed = np.array([-0.4,1.3])
        self.assertGreater(abs(np.clip(proposed,0,1).sum()-mass.sum()),0.01)
        out,_ = exchange(mass,np.ones(2),np.array([0]),np.array([1]),np.array([1.2]))
        self.assertAlmostEqual(out.sum(),mass.sum(),places=14)

    def test_gravity_and_capillary_sign(self):
        m = box_model((1,1,2),(0.01,0.01,0.02))
        t = Transport(m,Material(law='linear'))
        _,_,rate = t.rates(t.initial(0.5))
        self.assertLess(rate[0],0)  # edges point upward; water goes down.
        t = Transport(m,Material(law='linear'),gravity=(0,0,0))
        _,_,rate = t.rates(t.initial(np.array([0.8,0.2])))
        self.assertGreater(rate[0],0)  # wet -> dry.

    def test_hydrostatic_equilibrium(self):
        m = box_model((1,1,10),(0.01,0.01,0.1))
        p0 = 2000
        t = Transport(m,Material(law='linear',pressure_scale=p0))
        s = 0.8-1000*9.81*m.centers[:,2]/p0
        _,_,rates = t.rates(t.initial(s))
        np.testing.assert_allclose(rates,0,atol=1e-16)

    def test_boundary_budget_and_impermeable(self):
        m = box_model((2,1,2))
        for k in (0,1e-12):
            t = Transport(m,Material(permeability=k,law='linear'),face=boundary_face(m),feed_rate=1e-5)
            initial = t.initial(0,free=1e-5)
            final,history,_ = t.run(initial,2,max_dt=0.1)
            self.assertAlmostEqual(final.vector().sum(),initial.vector().sum(),places=14)
            self.assertTrue((final.pore <= t.capacity+1e-15).all())
            if k == 0:
                self.assertEqual(final.pore.sum(),0)
                self.assertGreater(final.surface.sum(),0)

    def test_zero_porosity(self):
        m = box_model((2,1,1),porosity=0)
        t = Transport(m,face=boundary_face(m),feed_rate=1e-5)
        state,_,_ = t.run(t.initial(free=1e-4),1,max_dt=0.1)
        self.assertEqual(state.pore.sum(),0)
        np.testing.assert_array_equal(t.saturation(state),0)

    def test_remap(self):
        a = box_model((3,1,1)); b = box_model((7,1,1))
        a.porosity[:] = [0.2,0.5,0.8]
        mass = a.capacity_mass*np.array([0.1,0.5,0.9])
        phi,new_mass,s = conservative_remap(a,b,mass)
        self.assertAlmostEqual(new_mass.sum(),mass.sum(),places=14)
        self.assertAlmostEqual(np.sum(b.volume*phi),a.pore_volume.sum(),places=14)
        self.assertTrue((s >= 0).all() and (s <= 1).all())

    def test_material_endpoints(self):
        pc,kr = Material().response(np.array([0.,0.5,1.]))
        self.assertTrue(np.isfinite(pc).all())
        self.assertTrue(pc[0] > pc[1] > pc[2])
        self.assertEqual(pc[-1],0)
        self.assertEqual(kr[0],0); self.assertEqual(kr[-1],1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
