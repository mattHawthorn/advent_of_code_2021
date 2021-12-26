from collections import Counter
from itertools import chain, islice, accumulate, repeat
from typing import Callable, Iterable, Mapping, TypeVar

T = TypeVar("T")


def step(pop: Mapping[int, int]) -> Mapping[int, int]:
    return dict(
        chain(
            ((i - 1, pop.get(i, 0)) for i in range(1, 7)),
            [
                (6, pop.get(0, 0) + pop.get(7, 0)),
                (7, pop.get(8, 0)),
                (8, pop.get(0, 0)),
            ],
        )
    )


def iterate(f: Callable[[T], T], x: T) -> Iterable[T]:
    xs = accumulate(repeat(f), lambda x, f: f(x), initial=x)
    return islice(xs, 1, None)


def simulate(population: Mapping[int, int], steps: int) -> Mapping[int, int]:
    pops = iterate(step, population)
    tail = islice(pops, steps - 1, None)
    return next(tail)


if __name__ == "__main__":
    import sys

    with open("day06.txt") if sys.stdin.isatty() else sys.stdin as f:
        counters = list(map(int, f.read().strip().split(",")))
        population = Counter(counters)

    final_pop = simulate(population, 80)
    print(sum(final_pop.values()))

    final_pop = simulate(population, 256)
    print(sum(final_pop.values()))
