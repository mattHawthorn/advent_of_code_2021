from collections import Counter
from functools import partial
from itertools import starmap
from operator import itemgetter
from typing import Callable, Iterable, Mapping, Tuple

# Objectives


def move_cost(pos1: int, pos2: int, count: int = 1) -> int:
    # part 1 cost function
    return count * abs(pos2 - pos1)


def move_cost2(pos1: int, pos2: int, count: int = 1) -> int:
    # part 2 cost function
    dist = abs(pos2 - pos1)
    return count * dist * (dist + 1) // 2


# Helpers


def total_move_cost(
    positions: Mapping[int, int], cost: Callable[[int, int, int], int], pos: int
) -> int:
    # sum of costs for all positions
    return sum(starmap(partial(cost, pos), positions.items()))


def ternary_search(start: int, stop: int, objective: Callable[[int], int]):
    if stop - start <= 2:
        mid = (start + stop) // 2
        l_ob, r_ob, m_ob = objective(start), objective(stop), objective(mid)
        return min([(start, l_ob), (mid, m_ob), (stop, r_ob)], key=itemgetter(1))

    l, r = (2 * start + stop) // 3, (start + 2 * stop) // 3
    l_ob, r_ob = objective(l), objective(r)
    if l_ob < r_ob:
        return ternary_search(start, r, objective)
    else:
        return ternary_search(l, stop, objective)


# Main function


def min_move_cost_ternary_search(
    positions: Iterable[int], cost: Callable[[int, int, int], int]
) -> Tuple[int, int]:
    counts = Counter(positions)
    min_, max_ = min(counts), max(counts)
    objective = partial(total_move_cost, counts, cost)
    return ternary_search(min_, max_, objective)


if __name__ == "__main__":
    import sys

    with open("day7.txt") if sys.stdin.isatty() else sys.stdin as f:
        positions = list(map(int, f.read().strip().split(",")))

    # Part 1
    print(min_move_cost_ternary_search(positions, move_cost))

    # Part 2
    print(min_move_cost_ternary_search(positions, move_cost2))
