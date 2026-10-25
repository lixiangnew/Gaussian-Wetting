"""M2: full Gaussian partition and positive deterministic quadrature (NumPy).

Project construction, inspired by partition-of-unity volume definitions; no
Darcy discretization or theorem for arbitrary Gaussian transport is claimed.
Coordinates are metres; measure_scale supplies cross-section (1D) / thickness
(2D), so all quadrature weights and resulting volumes are in m^3.
"""
from dataclasses import dataclass
from itertools import product
import numpy as np


@dataclass
class Nodes:
    centers: np.ndarray
    covariance: np.ndarray
    labels: np.ndarray

    def __post_init__(self):
        self.centers = np.array(self.centers, dtype=float, copy=True)
        self.covariance = np.array(self.covariance, dtype=float, copy=True)
        self.labels = np.array(self.labels, copy=True)
        if self.centers.ndim != 2 or not len(self.centers):
            raise ValueError('nonempty [N,d] centers required')
        n, d = self.centers.shape
        if d not in (1, 2, 3) or self.covariance.shape != (n, d, d):
            raise ValueError('invalid covariance dimensions')
        if self.labels.shape != (n,) or not np.issubdtype(self.labels.dtype, np.integer):
            raise ValueError('integer component labels required')
        if not np.isfinite(self.centers).all() or not np.isfinite(self.covariance).all():
            raise ValueError('finite geometry required')
        if not np.allclose(self.covariance, self.covariance.transpose(0, 2, 1), rtol=1e-12, atol=0):
            raise ValueError('covariance must be symmetric')
        self.cholesky = np.linalg.cholesky(self.covariance)
        self.inverse_cholesky = np.linalg.inv(self.cholesky)
        self.condition = np.linalg.cond(self.covariance)

    def weights(self, points, label):
        """Returns candidate indices, normalized weights, nearest Mahalanobis^2.

        Full support, peak amplitude a_i=1 (no determinant or volume prefactor).
        A missing component is an error. Distant points remain mathematically
        normalizable; their coverage distance must be diagnosed independently.
        """
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != self.centers.shape[1] or not np.isfinite(points).all():
            raise ValueError('finite [Q,d] points required')
        ids = np.flatnonzero(self.labels == label)
        if not len(ids):
            raise ValueError('component has no supporting nodes')
        delta = points[:, None, :] - self.centers[ids]
        whitened = np.einsum('nij,bnj->bni', self.inverse_cholesky[ids], delta, optimize=True)
        distance = np.einsum('bni,bni->bn', whitened, whitened)
        if not np.isfinite(distance).all():
            raise FloatingPointError('Mahalanobis distance overflow')
        nearest = distance.min(axis=1)
        logw = -0.5 * (distance - nearest[:, None])
        w = np.exp(logw)
        w /= w.sum(axis=1, keepdims=True)
        return ids, w, nearest


@dataclass(frozen=True)
class Box:
    lower: tuple
    upper: tuple
    label: int = 0
    measure_scale: float = 1.0

    def bounds(self):
        lo, hi = np.asarray(self.lower, float), np.asarray(self.upper, float)
        if lo.ndim != 1 or lo.shape != hi.shape or len(lo) not in (1, 2, 3):
            raise ValueError('invalid box dimension')
        if not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(hi <= lo):
            raise ValueError('invalid box bounds')
        if not np.isfinite(self.measure_scale) or self.measure_scale <= 0:
            raise ValueError('positive measure_scale required')
        if not isinstance(self.label, (int, np.integer)):
            raise ValueError('integer label required')
        return lo, hi

    @property
    def volume(self):
        lo, hi = self.bounds()
        return float(np.prod(hi-lo)*self.measure_scale)


def quadrature(box, cells, order=1, batch_size=128):
    """Composite Gauss-Legendre; order=1 is composite midpoint.

    Lazy tensor product: never materializes all Q*N weights or all Q points.
    """
    lo, hi = box.bounds()
    cells = np.broadcast_to(np.asarray(cells), lo.shape)
    if not np.issubdtype(cells.dtype, np.integer) or np.any(cells <= 0):
        raise ValueError('positive integer cells required')
    if not isinstance(order, (int, np.integer)) or order < 1 or not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError('positive integer order and batch size required')
    z, w = np.polynomial.legendre.leggauss(order)
    axes, weights = [], []
    for a, b, n in zip(lo, hi, cells):
        width = (b-a)/n
        axes.append((a+(np.arange(n)[:, None]+0.5)*width+0.5*width*z).ravel())
        weights.append(np.tile(0.5*width*w, n))
    shape = tuple(len(a) for a in axes)
    for start in range(0, int(np.prod(shape)), batch_size):
        ix = np.unravel_index(np.arange(start, min(start+batch_size, np.prod(shape))), shape)
        points = np.column_stack([a[j] for a, j in zip(axes, ix)])
        qw = np.prod(np.column_stack([a[j] for a, j in zip(weights, ix)]), axis=1)*box.measure_scale
        yield points, qw


def regular_nodes(box, shape, sigma_ratio=0.6):
    lo, hi = box.bounds()
    shape = np.broadcast_to(np.asarray(shape), lo.shape)
    if not np.issubdtype(shape.dtype, np.integer) or np.any(shape <= 0) or not np.isfinite(sigma_ratio) or sigma_ratio <= 0:
        raise ValueError('invalid node shape or sigma ratio')
    width = (hi-lo)/shape
    axes = [a+(np.arange(n)+0.5)*h for a, n, h in zip(lo, shape, width)]
    centers = np.array(list(product(*axes)))
    cov = np.broadcast_to(np.diag((sigma_ratio*width)**2), (len(centers), len(lo), len(lo))).copy()
    return Nodes(centers, cov, np.full(len(centers), box.label, int))


def integrate(nodes, boxes, cells, porosity=0.6, order=1, batch_size=128, density=1000.0):
    """Shared positive quadrature of volume and pore volume, independently by domain.

    porosity may be scalar or callable(points, label). Discontinuous material
    interfaces should be split into nonoverlapping integration boxes explicitly.
    """
    boxes = list(boxes)
    if not boxes or not np.isfinite(density) or density <= 0:
        raise ValueError('boxes and positive density required')
    bounds = [b.bounds() for b in boxes]
    for i, (lo, hi) in enumerate(bounds):
        if len(lo) != nodes.centers.shape[1]:
            raise ValueError('domain/node dimension mismatch')
        for old_lo, old_hi in bounds[:i]:
            if np.all(np.minimum(hi, old_hi) > np.maximum(lo, old_lo)):
                raise ValueError('integration boxes must not overlap')
    if set(nodes.labels.tolist()) != {b.label for b in boxes}:
        raise ValueError('node/domain components must match')
    inside = np.zeros(len(nodes.centers), dtype=bool)
    for box, (lo, hi) in zip(boxes, bounds):
        inside |= ((nodes.labels == box.label) & np.all(nodes.centers >= lo, axis=1)
                   & np.all(nodes.centers <= hi, axis=1))
    if not inside.all():
        raise ValueError('node centers must belong to their labeled physical domain')
    volume = np.zeros(len(nodes.centers)); pore = volume.copy()
    unity_error = 0.0; max_distance = 0.0; uncovered = 0; points_count = 0
    qtotal = 0.0; pore_total = 0.0; min_weight = 1.0
    for box in boxes:
        for points, qw in quadrature(box, cells, order, batch_size):
            ids, psi, distance = nodes.weights(points, box.label)
            phi = porosity(points, box.label) if callable(porosity) else porosity
            phi = np.broadcast_to(np.asarray(phi, float), (len(points),))
            if not np.isfinite(phi).all() or np.any((phi < 0) | (phi > 1)):
                raise ValueError('porosity must lie in [0,1]')
            volume[ids] += qw @ psi
            pore[ids] += (qw*phi) @ psi
            qtotal += float(qw.sum()); pore_total += float(qw @ phi)
            unity_error = max(unity_error, float(np.max(abs(psi.sum(axis=1)-1))))
            min_weight = min(min_weight, float(psi.min()))
            max_distance = max(max_distance, float(np.sqrt(distance.max())))
            uncovered += int(np.count_nonzero(distance > 9))
            points_count += len(points)
    if np.any(volume <= 0):
        raise FloatingPointError('zero numerical node volume: refine quadrature or inspect coverage')
    capacity = density*pore
    effective_phi = pore/volume
    analytical = sum(b.volume for b in boxes)
    metrics = dict(nodes=len(volume), quadrature_points=points_count,
                   volume_m3=float(volume.sum()), quadrature_volume_m3=qtotal,
                   analytic_volume_m3=analytical,
                   partition_total_relative_error=float(abs(volume.sum()-qtotal)/qtotal),
                   domain_quadrature_relative_error=abs(qtotal-analytical)/analytical,
                   capacity_kg=float(capacity.sum()),
                   capacity_partition_relative_error=float(abs(pore.sum()-pore_total)/max(pore_total, 1e-300)),
                   unity_max_error=unity_error, min_weight=min_weight,
                   min_volume_m3=float(volume.min()),
                   max_nearest_mahalanobis=max_distance,
                   fraction_beyond_3sigma=uncovered/points_count,
                   covariance_max_condition=float(nodes.condition.max()))
    return dict(volume=volume, pore_volume=pore, capacity=capacity,
                effective_porosity=effective_phi, metrics=metrics)
