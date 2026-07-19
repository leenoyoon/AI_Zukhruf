from collections import deque
from typing import List, Tuple, Optional

Point = Tuple[float, float]


def cross(o: Point, a: Point, b: Point) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def line_through(p: Point, q: Point) -> Tuple[float, float, float]:
    A = p[1] - q[1]
    B = q[0] - p[0]
    C = p[0] * q[1] - p[1] * q[0]
    return A, B, C


def eval_line(A: float, B: float, C: float, pt: Point) -> float:
    return A * pt[0] + B * pt[1] + C


class PathHalf:
    PUSH, POP_TOP, POP_BOT = "push", "pop_top", "pop_bot"

    def __init__(self, V: List[Point], tag: int, neighbour: int):
        self.V = V
        self.hull: deque = deque([neighbour, tag, neighbour])
        self.history: List[Tuple[str, int]] = [(self.PUSH, neighbour)]

    def _cross(self, o: int, a: int, b: int) -> float:
        return cross(self.V[o], self.V[a], self.V[b])

    def add(self, p: int) -> None:
        popped = False
        while len(self.hull) >= 3 and self._cross(self.hull[-2], self.hull[-1], p) <= 0:
            removed = self.hull.pop()
            self.history.append((self.POP_TOP, removed))
            popped = True
        while len(self.hull) >= 3 and self._cross(p, self.hull[0], self.hull[1]) <= 0:
            removed = self.hull.popleft()
            self.history.append((self.POP_BOT, removed))
            popped = True
        if popped:
            self.hull.append(p)
            self.hull.appendleft(p)
            self.history.append((self.PUSH, p))

    def split(self, target: int) -> None:
        while self.history and self.history[-1] != (self.PUSH, target):
            op, p = self.history.pop()
            if op == self.PUSH:
                self.hull.pop()
                self.hull.popleft()
            elif op == self.POP_TOP:
                self.hull.append(p)
            else:
                self.hull.appendleft(p)

    def find_extreme(self, A: float, B: float, C: float = 0.0) -> Tuple[int, float]:
        idxs = list(self.hull)
        m = len(idxs) - 1

        def proj(i: int) -> float:
            p = self.V[i]
            return A * p[0] + B * p[1] + C

        def brute() -> Tuple[int, float]:
            best = max(range(m), key=lambda i: abs(proj(idxs[i])))
            return idxs[best], abs(proj(idxs[best]))

        if m <= 6:
            return brute()

        def sign(a: int, b: int) -> bool:
            pa, pb = self.V[idxs[a]], self.V[idxs[b]]
            return (A * (pb[0] - pa[0]) + B * (pb[1] - pa[1])) >= 0

        lo, hi = 0, m - 1
        sbase = sign(hi, lo)
        brk = None
        guard = 0
        while lo < hi and guard <= m + 2:
            guard += 1
            brk = (lo + hi) // 2
            sbrk = sign(brk, brk + 1)
            if sbase == sbrk:
                if sbase == sign(lo, brk + 1):
                    lo = brk + 1
                else:
                    hi = brk
            else:
                break
        else:
            return brute()
        if brk is None:
            return brute()

        lo2, ml = lo, brk
        while lo2 < ml:
            mid = (lo2 + ml) // 2
            if sbase == sign(mid, mid + 1):
                lo2 = mid + 1
            else:
                ml = mid

        m2, hi2 = brk, m - 1
        while m2 < hi2:
            mid = (m2 + hi2) // 2
            if sbase == sign(mid, mid + 1):
                hi2 = mid
            else:
                m2 = mid + 1

        i1, i2 = idxs[lo2], idxs[m2]
        d1, d2 = abs(proj(i1)), abs(proj(i2))
        return (i1, d1) if d1 >= d2 else (i2, d2)


class DPHull:
    def __init__(self, V: List[Point], epsilon: float):
        if len(V) < 2:
            raise ValueError("need at least 2 points")
        self.V = V
        self.eps_sq = epsilon * epsilon
        self.tag: Optional[int] = None
        self.left: Optional[PathHalf] = None
        self.right: Optional[PathHalf] = None
        self.keep: List[bool] = [False] * len(V)

    def build(self, i: int, j: int) -> None:
        m = i + (j - i) // 2
        self.tag = m
        self.left = PathHalf(self.V, m, m - 1)
        for k in range(m - 2, i - 1, -1):
            self.left.add(k)
        self.right = PathHalf(self.V, m, m + 1)
        for k in range(m + 2, j + 1):
            self.right.add(k)

    def _dist_sq(self, A: float, B: float, C: float, idx: int) -> float:
        v = eval_line(A, B, C, self.V[idx])
        return v * v

    def dphull(self, i: int, j: int) -> int:
        if j - i <= 1:
            return j

        A, B, C = line_through(self.V[i], self.V[j])
        len_sq = A * A + B * B

        left, right, tag = self.left, self.right, self.tag
        lidx, _ = left.find_extreme(A, B, C)
        ridx, _ = right.find_extreme(A, B, C)
        ldist_sq = self._dist_sq(A, B, C, lidx)
        rdist_sq = self._dist_sq(A, B, C, ridx)

        if ldist_sq <= rdist_sq:
            if rdist_sq <= self.eps_sq * len_sq:
                return j
            if tag == ridx:
                self.build(i, ridx)
            else:
                right.split(ridx)
            self.keep[self.dphull(i, ridx)] = True
            self.build(ridx, j)
            return self.dphull(ridx, j)
        else:
            if ldist_sq <= self.eps_sq * len_sq:
                return j
            left.split(lidx)
            tmp = self.dphull(lidx, j)
            self.build(i, lidx)
            self.keep[self.dphull(i, lidx)] = True
            return tmp


    def dphull_iterative(self, i0: int, j0: int) -> None:
        stack: list = [("P", i0, j0)]
        while stack:
            item = stack.pop()
            if callable(item):
                item()
                continue
            _, i, j = item
            if j - i <= 1:
                continue

            A, B, C = line_through(self.V[i], self.V[j])
            len_sq = A * A + B * B
            left, right, tag = self.left, self.right, self.tag
            lidx, _ = left.find_extreme(A, B, C)
            ridx, _ = right.find_extreme(A, B, C)
            ldist_sq = self._dist_sq(A, B, C, lidx)
            rdist_sq = self._dist_sq(A, B, C, ridx)

            if ldist_sq <= rdist_sq:
                if rdist_sq <= self.eps_sq * len_sq:
                    continue
                if tag == ridx:
                    self.build(i, ridx)
                else:
                    right.split(ridx)

                def after_a(ridx=ridx, j=j):
                    self.keep[ridx] = True
                    self.build(ridx, j)
                    stack.append(("P", ridx, j))

                stack.append(after_a)
                stack.append(("P", i, ridx))
            else:
                if ldist_sq <= self.eps_sq * len_sq:
                    continue
                left.split(lidx)

                def after_c(i=i, lidx=lidx):
                    self.build(i, lidx)

                    def after_d(lidx=lidx):
                        self.keep[lidx] = True

                    stack.append(after_d)
                    stack.append(("P", i, lidx))

                stack.append(after_c)
                stack.append(("P", lidx, j))

    def run(self, use_iterative: bool = True) -> List[Point]:
        idx = self.run_indices(use_iterative=use_iterative)
        return [self.V[i] for i in idx]

    def run_indices(self, use_iterative: bool = True) -> List[int]:
        n = len(self.V)
        self.keep = [False] * n
        self.keep[0] = True
        self.build(0, n - 1)
        if use_iterative:
            self.dphull_iterative(0, n - 1)
            self.keep[n - 1] = True
        else:
            last = self.dphull(0, n - 1)
            self.keep[last] = True
        return [i for i in range(n) if self.keep[i]]


def simplify_dphull(points: List[Point], epsilon: float) -> List[Point]:
    if len(points) < 3:
        return list(points)
    return DPHull(points, epsilon).run()


def simplify_classic(points: List[Point], epsilon: float) -> List[Point]:
    n = len(points)
    if n < 3:
        return list(points)
    keep = [False] * n
    keep[0] = keep[n - 1] = True
    eps_sq = epsilon * epsilon

    def recurse(i: int, j: int) -> None:
        if j - i <= 1:
            return
        A, B, C = line_through(points[i], points[j])
        len_sq = A * A + B * B
        best_f, best_d = -1, -1.0
        for k in range(i + 1, j):
            d = eval_line(A, B, C, points[k]) ** 2
            if d > best_d:
                best_d, best_f = d, k
        if best_d > eps_sq * len_sq:
            keep[best_f] = True
            recurse(i, best_f)
            recurse(best_f, j)

    recurse(0, n - 1)
    return [points[i] for i in range(n) if keep[i]]


if __name__ == "__main__":
    import time
    import random
    import sys

    sys.setrecursionlimit(50_000)

    def zigzag_chain(n: int) -> List[Point]:
        return [(float(i), (1.0 if i % 2 == 0 else -1.0) * (1 + 1e-3 * i))
                for i in range(n)]

    print("Correctness sanity check on a small example:")
    demo_pts = zigzag_chain(20)
    out1 = simplify_dphull(demo_pts, 0.5)
    out2 = simplify_classic(demo_pts, 0.5)
    print(f"  dpHull kept {len(out1)} / {len(demo_pts)} points")
    print(f"  classic kept {len(out2)} / {len(demo_pts)} points")
    print()

    print("Timing on worst-case zig-zag chains:")
    print(f"  {'n':>7} {'classic O(n^2)':>15} {'dpHull O(n log n)':>18}")
    for n in (500, 1000, 2000, 4000, 8000):
        pts = zigzag_chain(n)
        t0 = time.perf_counter()
        simplify_classic(pts, 0.01)
        t1 = time.perf_counter()
        simplify_dphull(pts, 0.01)
        t2 = time.perf_counter()
        print(f"  {n:7d} {t1 - t0:15.4f} {t2 - t1:18.4f}")
    print()
    print("(classic time should roughly *quadruple* each time n doubles;")
    print(" dpHull time should roughly just *double*")