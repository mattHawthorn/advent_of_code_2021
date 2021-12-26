from itertools import starmap, tee
from operator import lt
from typing import Iterable, Tuple, TypeVar

T = TypeVar("T")


def windowed(n: int, it: Iterable[T]) -> Iterable[Tuple[T, ...]]:
    iters = tee(it, n)
    for i, it_ in enumerate(iters):
        for _ in range(i):
            next(it_)

    return zip(*iters)


if __name__ == "__main__":
    import sys

    with open("day01.txt") if sys.stdin.isatty() else sys.stdin as f:
        nums = list(map(int, filter(bool, map(str.strip, f))))

    # Part 1

    print(sum(starmap(lt, windowed(2, nums))))

    # Part 2

    print(sum(starmap(lt, windowed(2, map(sum, windowed(3, nums))))))
