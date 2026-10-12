"""针对体积、几何拓扑、容量和持久化的回归测试。"""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from model import Box, build_model


class RepresentationTests(unittest.TestCase):
    def test_resolution_invariance(self):
        for shape in [(1, 1, 1), (2, 3, 5), (8, 8, 8), (16, 16, 16)]:
            m = build_model([Box((0, 0, 0), (0.1, 0.06, 0.04), shape)])
            self.assertAlmostEqual(m.volume.sum(), 0.00024, places=15)
            self.assertAlmostEqual(m.capacity_mass.sum(), 0.144, places=14)
            self.assertTrue(np.all(m.volume > 0))
            self.assertTrue(np.all(np.linalg.eigvalsh(m.covariance) > 0))
            self.assertTrue(np.all(m.centers > m.cell_lower))
            self.assertTrue(np.all(m.centers < m.cell_upper))

    def test_graph_faces_and_edge_count(self):
        nx, ny, nz = 2, 3, 5
        m = build_model([Box((0, 0, 0), (0.1, 0.06, 0.04), (nx, ny, nz))])
        expected = (nx-1)*ny*nz + nx*(ny-1)*nz + nx*ny*(nz-1)
        self.assertEqual(len(m.edges), expected)
        self.assertEqual(len(np.unique(m.edges, axis=0)), expected)
        self.assertTrue(np.all(m.edges[:, 0] < m.edges[:, 1]))
        i, j = m.edges.T
        np.testing.assert_allclose(m.centers[j]-m.centers[i], m.edge_normal*m.center_distance[:, None])
        for e, (a, b) in enumerate(m.edges):
            d = int(np.argmax(m.edge_normal[e]))
            self.assertEqual(m.cell_upper[a, d], m.cell_lower[b, d])
            remaining = [k for k in range(3) if k != d]
            area = np.prod((m.cell_upper[a]-m.cell_lower[a])[remaining])
            self.assertAlmostEqual(m.face_area[e], area)
        self.assertEqual(m.summary()['components'], 1)

    def test_capacity_independent_of_covariance(self):
        box = Box((0, 0, 0), (1, 2, 3), (2, 3, 4))
        a, b = build_model([box], 0.25), build_model([box], 2.0)
        np.testing.assert_array_equal(a.capacity_mass, b.capacity_mass)
        np.testing.assert_array_equal(a.edges, b.edges)
        np.testing.assert_allclose(b.covariance, a.covariance*64)

    def test_air_gap_and_materials(self):
        m = build_model([Box((0, 0, 0), (1, 1, 1), (2, 2, 2), 0.2, 0.5),
                         Box((1.01, 0, 0), (2.01, 1, 1), (3, 3, 3), 0.8, 0.25)], 10)
        self.assertEqual(m.summary()['components'], 2)
        self.assertTrue(np.all(m.domain_id[m.edges[:, 0]] == m.domain_id[m.edges[:, 1]]))
        self.assertAlmostEqual(m.pore_volume.sum(), 1.0)
        self.assertAlmostEqual(m.water_mass.sum(), 300.0)

    def test_zero_porosity_and_singleton(self):
        m = build_model([Box((0, 0, 0), (1, 1, 1), (1, 1, 1), 0.0)])
        self.assertEqual(m.edges.shape, (0, 2))
        self.assertEqual(m.edge_normal.shape, (0, 3))
        self.assertEqual(m.capacity_mass.sum(), 0)
        self.assertEqual(m.summary()['components'], 1)

    def test_invalid_inputs(self):
        bad = [Box((0, 0, 0), (1, 1, 1), (0, 2, 2)),
               Box((0, 0, 0), (1, 1, 1), (2.5, 2, 2)),
               Box((0, 0, 0), (1, 1, 1), (2, 2, 2), 1.1),
               Box((0, 0, 0), (1, 1, 1), (2, 2, 2), 0, 0.1),
               Box((0, 0, 0), (0, 1, 1), (2, 2, 2)),
               Box((0, 0, 0), (float('nan'), 1, 1), (2, 2, 2))]
        for box in bad:
            with self.assertRaises(ValueError):
                build_model([box])
        box = Box((0, 0, 0), (1, 1, 1), (2, 2, 2))
        for args in [([],), ([box, box],), ([box], 0), ([box], 0.5, -1)]:
            with self.assertRaises(ValueError):
                build_model(*args)

    def test_save_roundtrip(self):
        m = build_model([Box((0, 0, 0), (1, 1, 1), (2, 3, 4))])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'model.npz'
            m.save(path)
            with np.load(path, allow_pickle=False) as data:
                for name in ('centers', 'covariance', 'volume', 'edges', 'porosity'):
                    np.testing.assert_array_equal(data[name], getattr(m, name))
                self.assertEqual(int(data['schema_version']), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
