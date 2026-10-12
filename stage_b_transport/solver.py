"""守恒有界显式图输运。仅适用于正交图和对角渗透率，默认 SI 单位。

主状态是质量，不在更新后 clip。全步/两个半步控制时间误差；不是通用
饱和承压 Richards 求解器，也不是任意非正交高斯图的准确通量离散。

state_t：当前水质量
    │
    ▼
rates(state_t)
    │
    ├─ saturation(state_t)：质量 → 饱和度
    │
    ├─ material.response(S)：饱和度 → 毛细压力、相对渗透率
    │
    └─ 加入重力与表面状态 → 计算各条边的请求流量
    │
    ▼
rates × dt：每秒流量 → 本步请求交换质量
    │
    ▼
exchange(...)
    │
    ├─ 检查供体库存
    ├─ 检查受体剩余容量
    └─ 同量扣除、同量增加
    │
    ▼
检查更新结果是否有限、是否越界
    │
    ▼
state_next：新的水质量
"""

from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Material:
    permeability: object = 1e-12  # scalar, [3], or [N,3], m²
    viscosity: float = 1e-3 # \(\mu\) 水动力黏度
    law: str = 'vg'
    pressure_scale: float = 1500.0  # VG 1/alpha (Pa); linear P0
    vg_n: float = 2.0
    saturation_floor: float = 0.02  # pressure regularization, not water creation

    def response(self, saturation):
        s = np.asarray(saturation, dtype=float)
        if self.law == 'linear':
            return self.pressure_scale * (1 - s), np.ones_like(s)
        if self.law != 'vg':
            raise ValueError('law must be linear or vg')
        # Rounding-level state deviations are tolerated in constitutive evaluation only.
        se = np.clip(s, 0, 1)
        eval_s = np.maximum(se, self.saturation_floor)
        m = 1 - 1 / self.vg_n
        pc = self.pressure_scale * np.maximum(eval_s ** (-1/m) - 1, 0) ** (1/self.vg_n)
        kr = np.sqrt(se) * (1 - (1 - se ** (1/m)) ** m) ** 2
        return pc, kr


@dataclass
class State:
    pore: np.ndarray             # kg at material nodes
    surface: np.ndarray          # kg at boundary reservoir patches
    free: float                 # kg in prescribed external supply inventory

    def vector(self):
        return np.concatenate((self.pore, self.surface, [self.free]))


def exchange(mass, capacity, first, second, requested):
    """同时限制有向质量交换。requested>0: first→second，负值反向。

    对每个节点聚合所有流出和流入，以库存/剩余容量计算比例。
    每条边取两端比例的较小值，因此不创造、不丢弃水，不依赖边顺序。
    本步新流入不能用于本步流出，新流出也不提前增加本步可接受容量。
    """
    donor = np.where(requested >= 0, first, second)
    receiver = np.where(requested >= 0, second, first)
    amount = np.abs(requested)
    outgoing = np.bincount(donor, weights=amount, minlength=len(mass))
    incoming = np.bincount(receiver, weights=amount, minlength=len(mass))
    donor_scale = np.ones_like(mass)
    receiver_scale = np.ones_like(mass)
    np.divide(np.maximum(mass, 0), outgoing, out=donor_scale, where=outgoing > 0)
    np.divide(np.maximum(capacity - mass, 0), incoming, out=receiver_scale, where=incoming > 0)
    factor = np.minimum(1, np.minimum(donor_scale[donor], receiver_scale[receiver]))
    # Tiny safety margin prevents arithmetic cancellation from exceeding a bound.
    transferred = amount * factor * (1 - 1e-14)
    delta = np.zeros_like(mass)
    np.add.at(delta, donor, -transferred)
    np.add.at(delta, receiver, transferred)
    return mass + delta, int(np.count_nonzero((factor < 1 - 1e-10) & (amount > 0)))


class Transport:
    def __init__(self, model, material=Material(), gravity=(0, 0, -9.81), face=None,
                 feed_rate=0.0):
        self.model, self.material = model, material
        self.n = len(model.volume)
        self.rho = model.water_density
        self.capacity = model.capacity_mass.copy()
        self.gravity = np.asarray(gravity, dtype=float)
        if self.gravity.shape != (3,) or not np.isfinite(self.gravity).all():
            raise ValueError('gravity must be a finite vector')
        if (not np.isfinite(material.viscosity) or material.viscosity <= 0 or
            not np.isfinite(material.pressure_scale) or material.pressure_scale <= 0 or
            not np.isfinite(material.vg_n) or material.vg_n <= 1 or
            not 0 < material.saturation_floor < 1):
            raise ValueError('invalid constitutive parameters')
        if material.law not in ('linear', 'vg'):
            raise ValueError('unsupported law')
        k = np.asarray(material.permeability, dtype=float)
        if k.ndim == 0:
            k = np.full((self.n, 3), float(k))
        elif k.shape == (3,):
            k = np.broadcast_to(k, (self.n, 3)).copy()
        elif k.shape != (self.n, 3):
            raise ValueError('permeability: scalar, [3], or [N,3] diagonal values only')
        if not np.isfinite(k).all() or (k < 0).any():
            raise ValueError('permeability must be finite and nonnegative')
        self.k = k.copy()
        self.k[self.capacity == 0] = 0
        self.i, self.j = model.edges.T
        normal = model.edge_normal
        ki = np.sum(self.k[self.i] * normal**2, axis=1)
        kj = np.sum(self.k[self.j] * normal**2, axis=1)
        harmonic = np.divide(2*ki*kj, ki+kj, out=np.zeros_like(ki), where=ki+kj > 0)
        self.conductance = self.rho * model.face_area / model.center_distance * harmonic / material.viscosity
        if face is None:
            self.nodes = np.empty(0, dtype=int)
            self.area = np.empty(0)
            self.position = np.empty((0, 3))
            self.half_distance = np.empty(0)
        else:
            self.nodes, self.area, self.position, self.half_distance = [np.asarray(a).copy() for a in face]
            if len(np.unique(self.nodes)) != len(self.nodes):
                raise ValueError('one patch per boundary node in this prototype')
            if (self.area <= 0).any() or (self.half_distance <= 0).any():
                raise ValueError('boundary geometry must be positive')
        self.b = len(self.nodes)
        self.feed_rate = float(feed_rate)
        if not np.isfinite(self.feed_rate) or self.feed_rate < 0:
            raise ValueError('feed_rate must be finite and nonnegative')
        if self.feed_rate > 0 and not self.b:
            raise ValueError('feed requires a boundary reservoir')
        displacement = model.centers[self.nodes] - self.position
        direction = np.divide(displacement, self.half_distance[:, None],
                              out=np.zeros_like(displacement), where=self.half_distance[:, None] > 0)
        kn = np.sum(self.k[self.nodes]*direction**2, axis=1)
        self.boundary_conductance = self.rho*self.area/self.half_distance*kn/material.viscosity

    def initial(self, saturation=0.0, free=0.0, surface=0.0):
        s = np.broadcast_to(np.asarray(saturation, dtype=float), (self.n,)).copy()
        if not np.isfinite(s).all() or (s < 0).any() or (s > 1).any():
            raise ValueError('invalid initial saturation')
        if not np.isfinite(free) or free < 0:
            raise ValueError('invalid free mass')
        surf = np.broadcast_to(np.asarray(surface, dtype=float), (self.b,)).copy()
        if not np.isfinite(surf).all() or (surf < 0).any():
            raise ValueError('invalid surface mass')
        return State(s*self.capacity, surf, float(free))

    def saturation(self, state):
        return np.divide(state.pore, self.capacity, out=np.zeros(self.n), where=self.capacity > 0)

    def rates(self, state):
        pc, kr = self.material.response(self.saturation(state))
        potential = -pc - self.rho*(self.model.centers @ self.gravity)
        rates = self.conductance * 0.5*(kr[self.i]+kr[self.j]) * (potential[self.i]-potential[self.j])
        first, second = self.i.copy(), self.j.copy()
        if self.b:
            h = state.surface / (self.rho*self.area)
            # h is reservoir-equivalent depth, not a resolved film shape.
            surface_potential = self.rho*np.linalg.norm(self.gravity)*h - self.rho*(self.position @ self.gravity)
            absorb = self.boundary_conductance * 0.5*(1+kr[self.nodes]) * np.maximum(surface_potential-potential[self.nodes], 0)
            patches = self.n + np.arange(self.b)
            first = np.concatenate((first, patches, np.full(self.b, self.n+self.b, dtype=int)))
            second = np.concatenate((second, self.nodes, patches))
            feed = self.feed_rate * self.area / self.area.sum()
            rates = np.concatenate((rates, absorb, feed))
        return first, second, rates

    def step(self, state, dt):
        first, second, rates = self.rates(state)
        capacity = np.concatenate((self.capacity, np.full(self.b+1, np.inf)))
        updated, limited = exchange(state.vector(), capacity, first, second, rates*dt)
        if not np.isfinite(updated).all():
            raise FloatingPointError('nonfinite mass')
        tolerance = max(float(capacity[:self.n].sum()), 1e-12)*1e-12
        if updated.min() < -tolerance or (updated[:self.n]-self.capacity).max() > tolerance:
            raise FloatingPointError('capacity violation')
        return State(updated[:self.n], updated[self.n:self.n+self.b], float(updated[-1])), limited

    def run(self, state, end_time, max_dt=0.5, tolerance=2e-5, max_steps=200000):
        """比较一个全步和两个半步；接受更准确的两个半步结果。"""
        if not np.isfinite(end_time) or end_time < 0 or max_dt <= 0 or tolerance <= 0:
            raise ValueError('invalid integration controls')
        initial_total = float(state.vector().sum())
        pore_scale = np.maximum(self.capacity, 1e-18)
        surface_scale = np.maximum(self.capacity[self.nodes], 1e-12)
        scale = np.concatenate((pore_scale, surface_scale, [max(initial_total, 1e-12)]))
        time, dt, accepted, rejected, limited_total = 0.0, max_dt, 0, 0, 0
        history = []
        def record():
            s = self.saturation(state)
            history.append({'time_s': time, 'free_kg': state.free,
                            'surface_kg': float(state.surface.sum()), 'pore_kg': float(state.pore.sum()),
                            'mass_residual_kg': float(state.vector().sum()-initial_total),
                            'min_s': float(s.min()), 'max_s': float(s.max()),
                            'max_h_m': float((state.surface/(self.rho*self.area)).max()) if self.b else 0.0})
        record()
        for _ in range(max_steps):
            if time >= end_time:
                break
            dt = min(dt, max_dt, end_time-time)
            full, _ = self.step(state, dt)
            half, limited1 = self.step(state, dt/2)
            refined, limited2 = self.step(half, dt/2)
            error = float(np.max(np.abs(full.vector()-refined.vector())/scale))
            if error > tolerance:
                rejected += 1
                dt *= max(0.1, 0.8*np.sqrt(tolerance/error))
                if dt < 1e-12:
                    raise RuntimeError('time step underflow; revise constitutive regularization')
                continue
            state = refined
            time = min(end_time, time+dt)
            accepted += 1
            limited_total += limited1+limited2
            record()
            dt *= min(2.0, 0.9*np.sqrt(tolerance/max(error, 1e-30)))
        else:
            raise RuntimeError('max_steps exceeded')
        return state, history, {'accepted_steps': accepted, 'rejected_steps': rejected,
                                'limited_edge_updates': limited_total,
                                'min_accepted_dt_s': min(np.diff([r['time_s'] for r in history]), default=0.0),
                                'max_accepted_dt_s': max(np.diff([r['time_s'] for r in history]), default=0.0),
                                'max_mass_residual_kg': max(abs(r['mass_residual_kg']) for r in history)}
