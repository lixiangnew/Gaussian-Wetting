import unittest
import numpy as np
from partition import Box, Nodes, regular_nodes, integrate, quadrature


class PartitionTests(unittest.TestCase):
    def test_single_node_and_physical_units(self):
        for d, scale in [(1, 0.0001), (2, 0.01), (3, 1.0)]:
            box = Box((0,)*d, (0.1,)*d, measure_scale=scale)
            n = regular_nodes(box, 1)
            r = integrate(n, [box], 4, porosity=0.4)
            np.testing.assert_allclose(r['volume'], [box.volume], rtol=1e-14)
            np.testing.assert_allclose(r['capacity'], [400*box.volume], rtol=1e-14)

    def test_two_node_symmetry_and_constant(self):
        b = Box((0,), (1,)); n = regular_nodes(b, 2)
        r = integrate(n, [b], 100)
        np.testing.assert_allclose(r['volume'], [0.5, 0.5], atol=1e-14)
        _, w, _ = n.weights(np.array([[0.0], [0.3], [1.0]]), 0)
        np.testing.assert_allclose(w @ np.full(2, 0.37), 0.37, atol=1e-15)
        self.assertTrue((w >= 0).all())

    def test_far_points_and_missing_support(self):
        n = regular_nodes(Box((0,), (1,)), 3)
        _, w, dist = n.weights(np.array([[1000.0]]), 0)
        self.assertTrue(np.isfinite(w).all())
        self.assertAlmostEqual(w.sum(), 1)
        self.assertGreater(dist[0], 9)
        with self.assertRaises(ValueError):
            n.weights(np.array([[0.5]]), 9)

    def test_rotated_spd_kernel(self):
        c = np.array([[0., 0.]])
        cov = np.array([[[2., 0.5], [0.5, 1.]]])
        n = Nodes(c, cov, [0])
        p = np.array([[1., 2.]])
        _, _, d = n.weights(p, 0)
        self.assertAlmostEqual(d[0], float(p[0] @ np.linalg.inv(cov[0]) @ p[0]))

    def test_reject_invalid_inputs(self):
        with self.assertRaises(np.linalg.LinAlgError):
            Nodes([[0.]], [[[-1.]]], [0])
        with self.assertRaises(ValueError):
            Nodes([[0., 0.]], [[[1., 1.], [0., 1.]]], [0])
        b = Box((0,), (1,)); n = regular_nodes(b, 2)
        for kwargs in [dict(porosity=-0.1), dict(cells=0), dict(batch_size=0)]:
            args = dict(cells=10); args.update(kwargs)
            with self.assertRaises(ValueError):
                integrate(n, [b], **args)
        with self.assertRaises(ValueError):
            integrate(n, [b, b], 10)
        outside = Nodes([[2.]], [[[0.1]]], [0])
        with self.assertRaises(ValueError):
            integrate(outside, [b], 10)

    def test_domain_isolation(self):
        boxes = [Box((0,), (1,), 0), Box((1.01,), (2.01,), 1)]
        ns = [regular_nodes(b, 3, sigma_ratio=10) for b in boxes]
        n = Nodes(np.concatenate([a.centers for a in ns]), np.concatenate([a.covariance for a in ns]),
                  np.concatenate([a.labels for a in ns]))
        r = integrate(n, boxes, 40, porosity=lambda x, label: 0.2 if label == 0 else 0.8)
        for label, phi in [(0, 0.2), (1, 0.8)]:
            self.assertAlmostEqual(r['volume'][n.labels == label].sum(), 1)
            self.assertAlmostEqual(r['capacity'][n.labels == label].sum(), 1000*phi)
        ids, _, _ = n.weights(np.array([[0.99]]), 0)
        self.assertTrue((n.labels[ids] == 0).all())

    def test_nonuniform_capacity_and_zero_porosity(self):
        b = Box((0,), (1,)); n = regular_nodes(b, 5)
        pieces = [Box((0,), (0.37,)), Box((0.37,), (1,))]
        r = integrate(n, pieces, 32, porosity=lambda x, _: np.where(x[:, 0] < 0.37, 0.2, 0.8))
        self.assertAlmostEqual(r['capacity'].sum(), 1000*(0.37*0.2+0.63*0.8), places=10)
        z = integrate(n, [b], 20, porosity=0)
        np.testing.assert_array_equal(z['capacity'], 0)

    def test_batch_invariance(self):
        b = Box((0, 0), (1, 1)); n = regular_nodes(b, (3, 4))
        a = integrate(n, [b], 12, batch_size=1)
        c = integrate(n, [b], 12, batch_size=127)
        np.testing.assert_allclose(a['volume'], c['volume'], rtol=2e-14)

    def test_midpoint_convergence(self):
        b = Box((0,), (1,)); n = regular_nodes(b, 5, 0.8)
        n.centers[1, 0] += 0.04
        reference = integrate(n, [b], 128, order=4)['volume']
        errors = [np.linalg.norm(integrate(n, [b], q)['volume']-reference) for q in (16, 32, 64)]
        self.assertLess(errors[2], errors[0]/8)

    def test_positive_gauss_rule_integrates_polynomial(self):
        b = Box((0,), (1,), measure_scale=0.01)
        total = 0
        for p, w in quadrature(b, 2, order=3):
            self.assertTrue((w > 0).all())
            total += w @ p[:, 0]**4
        self.assertAlmostEqual(total, 0.01/5, places=15)


if __name__ == '__main__':
    unittest.main(verbosity=2)
