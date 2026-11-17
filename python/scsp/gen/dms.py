"""Synthetic DMS-type set covering instances.

The model has the same shape as the one built in the DMS competition code
(`HighsSP.py`, read-only reference):

    min  sum_i (c_i / K) x_i
    s.t. sum_{i contains k} x_i >= 1            for every order k
         sum_{i : type(i) = t} x_i <= avail_t    for t in WALK, BIKE, CAR
         x binary

A column is a bundle: a rider type plus a pickup sequence and a delivery
sequence over a few orders (all pickups first, then all deliveries). Its cost is
`fixed_cost + route_metres / 100 * var_cost`, as in the DMS `Rider.calculate_cost`.

What is synthetic here: order coordinates, volumes, time windows and the way the
bundle pool is sampled (random neighbourhoods instead of an ALNS history). Rider
parameters are set at the level of the OGC 2024 example problems; they are not a
copy of any competition file. Every instance is feasible by construction: a
hidden solution that respects rider availability is put into the pool and its
column indices are stored in meta["warm_start"].
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..model import COVER, PACK, Model

RIDER_TYPES = ("WALK", "BIKE", "CAR")       # same indices as RIDER_DICT in HighsSP.py


@dataclass(frozen=True)
class RiderSpec:
    speed: float            # metres per second
    capacity: float
    var_cost: float         # per 100 m
    fixed_cost: float
    service_time: float     # seconds added to every leg


DEFAULT_RIDERS = (
    RiderSpec(speed=1.32, capacity=70, var_cost=30, fixed_cost=8000, service_time=300),    # WALK
    RiderSpec(speed=5.29, capacity=100, var_cost=60, fixed_cost=8000, service_time=120),   # BIKE
    RiderSpec(speed=4.23, capacity=200, var_cost=100, fixed_cost=6000, service_time=180),  # CAR
)
DEFAULT_AVAIL_FRAC = (0.30, 0.20, 1.00)     # available riders per order, by type
DEFAULT_SIZE_PROB = (0.15, 0.33, 0.30, 0.16, 0.06)   # bundle size 1..5


@dataclass(frozen=True)
class DmsParams:
    n_orders: int
    n_bundles: int
    seed: int = 1
    tight: bool = False             # availability = 1.05 x usage of the hidden solution
    area: float = 6000.0            # side of the square region, metres
    dlv_sigma: float = 1200.0       # spread of delivery points around their shop
    road_factor: float = 1.25       # euclidean -> road distance
    knn: int = 12                   # neighbourhood size used to sample bundle partners
    dominated_frac: float = 0.2     # share of bundles that keep a random (not the best) route


class _World:
    """Orders, distances and route evaluation for one instance."""

    def __init__(self, p: DmsParams, rng: np.random.Generator, riders=DEFAULT_RIDERS):
        K = p.n_orders
        self.K, self.riders = K, riders
        n_shops = max(5, K // 4)
        centres = rng.uniform(0, p.area, size=(n_shops, 2))
        shop = centres[rng.integers(0, n_shops, size=K)] + rng.normal(0, 60, size=(K, 2))
        dlv = shop + rng.normal(0, p.dlv_sigma, size=(K, 2))
        pts = np.clip(np.vstack([shop, dlv]), 0, p.area)        # 0..K-1 shops, K..2K-1 deliveries
        diff = pts[:, None, :] - pts[None, :, :]
        self.dist = np.rint(np.sqrt((diff ** 2).sum(-1)) * p.road_factor)
        self.volume = rng.integers(10, 61, size=K).astype(float)
        horizon = 1800.0 + 18.0 * K
        order_time = np.sort(rng.uniform(0, horizon, size=K))
        self.ready = np.rint(order_time + rng.uniform(600, 1200, size=K))
        car = riders[2]
        direct = np.rint(self.dist[np.arange(K), np.arange(K) + K] / car.speed + car.service_time)
        self.deadline = self.ready + direct + np.rint(rng.uniform(1500, 3600, size=K))
        self.travel = [np.rint(self.dist / r.speed + r.service_time) for r in riders]
        # neighbourhood metric: shops close, deliveries close, ready times close
        d_shop = self.dist[:K, :K]
        d_dlv = self.dist[K:, K:]
        d_time = np.abs(self.ready[:, None] - self.ready[None, :]) * riders[1].speed
        score = d_shop + d_dlv + 0.5 * d_time
        np.fill_diagonal(score, np.inf)
        k = min(p.knn, K - 1)
        self.neigh = np.argsort(score, axis=1)[:, :k]

    def route(self, t: int, src: tuple, dst: tuple):
        """(feasible, metres) of rider type t doing pickups `src` then deliveries `dst`."""
        K, T, D = self.K, self.travel[t], self.dist
        now = self.ready[src[0]]
        metres = 0.0
        prev = src[0]
        for o in src[1:]:
            now = max(now + T[prev, o], self.ready[o])
            metres += D[prev, o]
            prev = o
        for o in dst:
            node = o + K
            now += T[prev, node]
            metres += D[prev, node]
            if now > self.deadline[o]:
                return False, metres
            prev = node
        return True, metres

    def _orderings(self, orders: list, rng: np.random.Generator):
        K, D = self.K, self.dist
        by_ready = tuple(sorted(orders, key=lambda o: self.ready[o]))
        by_dead = tuple(sorted(orders, key=lambda o: self.deadline[o]))
        cands = {(by_ready, by_dead)}
        # nearest-neighbour pickups from the earliest-ready shop, nearest-neighbour deliveries after
        left, src = set(orders) - {by_ready[0]}, [by_ready[0]]
        while left:
            nxt = min(left, key=lambda o: D[src[-1], o])
            src.append(nxt)
            left.remove(nxt)
        left, dst, prev = set(orders), [], src[-1]
        while left:
            nxt = min(left, key=lambda o: D[prev, o + K])
            dst.append(nxt)
            left.remove(nxt)
            prev = nxt + K
        cands.add((tuple(src), tuple(dst)))
        cands.add((by_ready, tuple(dst)))
        cands.add((tuple(src), by_dead))
        if len(orders) > 1:
            cands.add((tuple(rng.permutation(orders).tolist()), tuple(rng.permutation(orders).tolist())))
        return sorted(cands)

    def best_bundle(self, t: int, orders: list, rng: np.random.Generator, pick_random: bool = False):
        """Best (or a random) feasible route for the order set, or None."""
        if self.volume[orders].sum() > self.riders[t].capacity:
            return None
        feas = []
        for src, dst in self._orderings(orders, rng):
            ok, metres = self.route(t, src, dst)
            if ok:
                feas.append((metres, src, dst))
        if not feas:
            return None
        metres, src, dst = feas[int(rng.integers(len(feas)))] if pick_random else min(feas)
        r = self.riders[t]
        return (t, src, dst, r.fixed_cost + metres / 100.0 * r.var_cost)


def _hidden_solution(w: _World, rng: np.random.Generator, avail: np.ndarray, size_prob) -> list:
    """Greedy feasible assignment of every order to one bundle within rider availability."""
    K = w.K
    left = avail.astype(np.int64).copy()
    free = np.ones(K, dtype=bool)
    sol = []
    for i in rng.permutation(K):
        if not free[i]:
            continue
        chosen = None
        size = int(rng.choice(len(size_prob), p=size_prob)) + 1
        for s in range(size, 0, -1):
            partners = [o for o in w.neigh[i] if free[o]][: s - 1]
            if len(partners) < s - 1:
                continue
            orders = [int(i)] + [int(o) for o in partners]
            for t in rng.permutation(3):
                if left[t] <= 0:
                    continue
                b = w.best_bundle(int(t), orders, rng)
                if b is not None:
                    chosen = b
                    break
            if chosen is not None:
                break
        if chosen is None:
            raise RuntimeError("no rider left for the hidden solution; raise availability")
        left[chosen[0]] -= 1
        for o in chosen[1]:
            free[o] = False
        sol.append(chosen)
    return sol


def generate_dms(
    n_orders: int,
    n_bundles: int,
    seed: int = 1,
    tight: bool = False,
    name: str | None = None,
    riders=DEFAULT_RIDERS,
    avail_frac=DEFAULT_AVAIL_FRAC,
    size_prob=DEFAULT_SIZE_PROB,
    **overrides,
) -> Model:
    p = DmsParams(n_orders=n_orders, n_bundles=n_bundles, seed=seed, tight=tight, **overrides)
    K = p.n_orders
    if K < 4:
        raise ValueError("need at least 4 orders")
    rng = np.random.default_rng([p.seed, K, p.n_bundles, int(p.tight)])
    w = _World(p, rng, riders)
    size_prob = np.asarray(size_prob, dtype=float) / np.sum(size_prob)

    loose = np.ceil(np.asarray(avail_frac, dtype=float) * K).astype(np.int64)
    hidden = _hidden_solution(w, rng, loose, size_prob)
    if len(hidden) > p.n_bundles:
        raise ValueError(f"n_bundles={p.n_bundles} is smaller than the hidden solution ({len(hidden)} bundles)")
    used = np.bincount([b[0] for b in hidden], minlength=3)
    avail = np.maximum(1, np.ceil(1.05 * used)).astype(np.int64) if p.tight else loose

    pool: dict = {}
    for b in hidden:
        pool[(b[0], b[1], b[2])] = b[3]
    hidden_keys = list(pool)
    attempts, max_attempts = 0, 60 * p.n_bundles + 1000
    while len(pool) < p.n_bundles and attempts < max_attempts:
        attempts += 1
        i = int(rng.integers(K))
        size = int(rng.choice(len(size_prob), p=size_prob)) + 1
        orders = [i]
        if size > 1:
            orders += [int(o) for o in rng.choice(w.neigh[i], size=min(size - 1, w.neigh.shape[1]), replace=False)]
        b = w.best_bundle(int(rng.integers(3)), orders, rng, pick_random=rng.random() < p.dominated_frac)
        if b is not None:
            pool.setdefault((b[0], b[1], b[2]), b[3])
    if len(pool) < p.n_bundles:
        raise RuntimeError(f"only {len(pool)} distinct feasible bundles found (wanted {p.n_bundles})")

    # shuffle so that the hidden solution is not the first block of columns
    keys = list(pool)
    perm = rng.permutation(len(keys))
    keys = [keys[j] for j in perm]
    index_of = {k: j for j, k in enumerate(keys)}
    col_type = np.array([k[0] for k in keys], dtype=np.int64)
    bundle_cost = np.array([pool[k] for k in keys], dtype=np.float64)

    sizes = np.array([len(k[1]) for k in keys], dtype=np.int64)
    col_ptr = np.zeros(len(keys) + 1, dtype=np.int64)
    np.cumsum(sizes, out=col_ptr[1:])
    row_idx = np.fromiter((o for k in keys for o in k[1]), dtype=np.int64, count=int(col_ptr[-1]))
    side_cols = [np.flatnonzero(col_type == t).astype(np.int32) for t in range(3)]
    side_ptr = np.zeros(4, dtype=np.int64)
    np.cumsum([len(c) for c in side_cols], out=side_ptr[1:])

    warm = sorted(index_of[k] for k in hidden_keys)
    name = name or f"dms_K{K}_n{p.n_bundles}{'_tight' if p.tight else ''}_s{p.seed}"
    return Model.from_columns(
        name, bundle_cost / K, K, col_ptr, row_idx, COVER,
        side_ptr=side_ptr, side_idx=np.concatenate(side_cols), side_val=np.ones(int(side_ptr[-1])),
        side_sense=np.full(3, PACK, dtype="U1"), side_rhs=avail.astype(np.float64),
        meta={
            "format": "synthetic-dms", "params": asdict(p), "rider_types": list(RIDER_TYPES),
            "riders": [asdict(r) for r in riders], "avail": avail.tolist(),
            "hidden_usage": used.tolist(), "warm_start": warm,
            "warm_start_obj": float(bundle_cost[warm].sum() / K), "cost_scale": f"bundle cost / K (K={K})",
        },
    )
