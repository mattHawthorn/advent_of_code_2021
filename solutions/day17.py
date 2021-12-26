from functools import partial
from itertools import product
from math import ceil, floor, sqrt
from typing import Iterable, Iterator, Optional, Tuple

Coord = Tuple[int, int]
Velocity = Tuple[int, int]


class Range(tuple):
    def __new__(cls, *args):
        return super().__new__(cls, sorted(args))


def vs_hitting_target(x_range: Range, y_range: Range) -> Iterator[Velocity]:
    vs = product(xvs_can_hit(*x_range), yvs_can_hit(*y_range))
    hits = partial(hits_target, x_range, y_range)
    return filter(hits, vs)


def hits_target(x_range: Range, y_range: Range, velocity: Velocity) -> bool:
    xv, yv = velocity
    enter_exit_x = enter_exit_time_x(xv, *x_range)
    if enter_exit_x is None:
        return False
    t1x, t2x = enter_exit_x
    for enter_exit_y in enter_exit_time_y(yv, *y_range):
        t1y, t2y = enter_exit_y
        if max(ceil_(t1x), ceil_(t1y)) <= min(floor_(t2x), floor(t2y)):
            return True
    return False


def xvs_can_hit(xmin: int, xmax: int) -> Iterable[int]:
    if xmin <= 0 and xmax <= 0:
        for xv in xvs_can_hit(-xmax, -xmin):
            yield -xv
    else:
        if xmin <= 0:
            xvmin = xmin
        else:
            xvmins = quadratic(1, 1, -2 * xmin)
            xvmin = (
                None if xvmins is None else int(ceil(min(v for v in xvmins if v > 0)))
            )
        if xvmin is not None:
            yield from range(xvmin, xmax + 1)


def yvs_can_hit(ymin: int, ymax: int) -> Iterable[int]:
    if ymin >= 0:
        yvmins = quadratic(1, 1, -2 * ymin)
        if yvmins is not None:
            yvmin = int(ceil(min(v for v in yvmins if v > 0)))
            yield from range(yvmin, ymax + 1)
    else:
        # how fast we'll be going right after passing back through 0
        yvmax = abs(ymin) - 1 if ymax < 0 else ymax
        yield from range(ymin, yvmax + 1)


def enter_exit_time_y(yv0, ymin, ymax) -> Iterator[Tuple[float, float]]:
    # all above x axis
    ts1 = t_of_y(yv0, ymin)
    ts2 = t_of_y(yv0, ymax)
    if ts1 is not None and ts2 is not None:
        t_enter1, t_exit2 = sorted(ts1)
        t_exit1, t_enter2 = sorted(ts2)
        yield t_enter1, t_exit1
        yield t_enter2, t_exit2
    elif ts1 is not None:
        t_enter1, t_exit1 = ts1
        yield t_enter1, t_exit1


def enter_exit_time_x(xv0, xmin, xmax) -> Optional[Tuple[float, float]]:
    t1 = t_of_x(xv0, xmax if xv0 <= 0 else xmin)
    t2 = t_of_x(xv0, xmin if xv0 <= 0 else xmax)
    if t1 is None:
        return None
    elif t2 is None:
        return t1, float("inf")
    else:
        return t1, t2


# Helpers

inf = float("inf")


def ceil_(x):
    if x == inf:
        return x
    return ceil(x)


def floor_(x):
    if x == inf:
        return x
    return floor(x)


def sign(x: int) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)


def y_max(yv: int) -> int:
    if yv <= 0:
        return 0
    return yv * (yv + 1) // 2


def quadratic(a, b, c) -> Optional[Tuple[float, float]]:
    d = b * b - 4 * a * c
    if d < 0:
        return None
    sq = sqrt(d)
    denom = 2 * a
    return (-b - sq) / denom, (-b + sq) / denom


def t_of_x(xv0: int, x: int) -> Optional[float]:
    """Time to reach horizontal position x"""
    if x == 0:
        return None
    direction = sign(xv0)
    if direction != 0:
        x *= direction
        xv0 *= direction
    x_max = xv0 * (xv0 + 1) / 2
    if x > x_max or x < 0:
        return None
    ts = quadratic(1 / 2, -xv0 - 1 / 2, x)
    if ts is None:
        return None
    return min(t for t in ts if t >= 0)


def t_of_y(yv0: int, y: int) -> Optional[Tuple[float, float]]:
    """Time to reach vertical position y - 2 times as there are 2 possible solutions"""
    return quadratic(1 / 2, -yv0 - 1 / 2, y)


# Tests


def test():
    def ballistic_path(velocity: Velocity) -> Iterator[Tuple[Coord, Coord]]:
        x, y = 0, 0
        xv, yv = velocity
        while True:
            yield (x, y), (xv, yv)
            x += xv
            y += yv
            yv -= 1
            xv -= sign(xv)

    def x_of_t(xv0, t):
        t = min(t, xv0 + 1)
        return t * xv0 + t * (1 - t) // 2

    def y_of_t(xv0, t):
        return t * xv0 + t * (1 - t) // 2

    def x_max(xv: int) -> int:
        v = abs(xv)
        return sign(xv) * v * (v + 1) // 2

    print("running tests")

    xv = 6
    yv = 9
    v = (xv, yv)
    assert x_max(xv) == 21
    x_max_t = t_of_x(xv, x_max(xv))
    assert x_max_t == 6
    for t, (xy, v) in zip(range(30), ballistic_path(v)):
        x = x_of_t(xv, t)
        y = y_of_t(yv, t)
        if 0 < t <= x_max_t:
            assert t_of_x(xv, x) == t, (t_of_x(xv, x), t)
        assert t in t_of_y(yv, y), (t_of_y(yv, y), t)
        assert xy == (x, y)

    assert set(xvs_can_hit(7, 20)) == set(range(4, 21))
    assert set(yvs_can_hit(11, 30)) == set(range(5, 31))
    assert set(yvs_can_hit(-10, 15)) == set(range(-10, 16))
    assert set(yvs_can_hit(-20, -10)) == set(range(-20, 20))
    assert set(yvs_can_hit(-10, -5)) == set(range(-10, 10))

    xr, yr = Range(20, 30), Range(-10, -5)
    for v in (7, 2), (6, 3), (9, 0):
        assert hits_target(xr, yr, v)

    vs = list(vs_hitting_target(xr, yr))
    assert y_max(max(yv for xv, yv in vs)) == 45
    assert len(vs) == 112


if __name__ == "__main__":
    import re
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test()

    # Parse input

    with open("day17.txt") if sys.stdin.isatty() else sys.stdin as f:
        inputstr = f.read()
        xmin, xmax, ymin, ymax = map(int, re.findall(r"-?\d+", inputstr))
        xrange = Range(xmin, xmax)
        yrange = Range(ymin, ymax)

    # Part 1

    vs = list(vs_hitting_target(xrange, yrange))
    print(y_max(max(yv for xv, yv in vs)))

    # Part 2

    print(len(vs))
