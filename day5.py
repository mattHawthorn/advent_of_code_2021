from itertools import accumulate, repeat, chain
from typing import Counter, Iterable, List, Tuple

Coord = Tuple[int, int]
LineSpec = Tuple[Coord, Coord]


def gen_points(line: LineSpec) -> Iterable[Coord]:
    (x1, y1), (x2, y2) = line
    x_diff = x2 - x1
    y_diff = y2 - y1
    num_steps = gcd(abs(x_diff), abs(y_diff))
    step = x_diff // num_steps, y_diff // num_steps
    return accumulate(repeat(step, num_steps), add_coords, initial=(x1, y1))


def add_coords(c1: Coord, c2: Coord) -> Coord:
    (x1, y1), (x2, y2) = c1, c2
    return x1 + x2, y1 + y2


def gcd(m, n):
    # euclid
    larger, smaller = (m, n) if m > n else (n, m)
    if smaller == 0:
        return larger
    _, rem = divmod(larger, smaller)
    return gcd(smaller, rem)


def coord_counts(lines: Iterable[LineSpec]) -> Counter[Coord]:
    coords = chain.from_iterable(map(gen_points, lines))
    return Counter(coords)


def crowded_coords(counts: Counter[Coord], max_count: int = 1) -> List[Coord]:
    return [c for c, n in counts.items() if n > max_count]


def is_axis_aligned(line: LineSpec) -> bool:
    (x1, y1), (x2, y2) = line
    return x1 == x2 or y1 == y2


# Input parsing


def parse_line(s: str) -> LineSpec:
    l, r = s.strip().split(" -> ")
    return parse_coord(l), parse_coord(r)


def parse_coord(s: str) -> Coord:
    x, y = s.strip().split(",")
    return int(x), int(y)


if __name__ == "__main__":
    import sys

    with open("day5.txt") if sys.stdin.isatty() else sys.stdin as f:
        lines = list(map(parse_line, iter(f.readline, "")))

    counts = coord_counts(filter(is_axis_aligned, lines))
    crowded = crowded_coords(counts)
    print(len(crowded), crowded[:10])

    counts = coord_counts(lines)
    crowded = crowded_coords(counts)
    print(len(crowded), crowded[:10])
