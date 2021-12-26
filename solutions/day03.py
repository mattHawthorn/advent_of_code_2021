from collections import Counter
from functools import partial
from itertools import accumulate, repeat, takewhile
from operator import eq, itemgetter, mul
from typing import (
    Callable,
    Hashable,
    Iterable,
    Iterator,
    List,
    Sequence,
    Tuple,
    TypeVar,
)

T = TypeVar("T", bound=Hashable)


def frequency_extremes(
    it: Iterable[T], *reducers: Callable[[Iterable[Tuple[T, int]]], T]
) -> Tuple[T, ...]:
    counts = Counter(it)
    items = counts.items()
    return tuple(f(items)[0] for f in reducers)


def least_and_most_common(it: Iterable[T]) -> Tuple[T, T]:
    min_, max_ = frequency_extremes(it, min_by_count, max_by_count)
    return min_, max_


def iterate(f: Callable[[T], T], x: T) -> Iterator[T]:
    return accumulate(repeat(f), lambda x, f: f(x), initial=x)


def filter_by_frequency(
    seqs: Sequence[Sequence[T]], reducer: Callable[[Iterable[Tuple[T, int]]], T]
) -> Sequence[T]:
    def filter_(
        ix_seqs: Tuple[int, Sequence[Sequence[T]]]
    ) -> Tuple[int, List[Sequence[T]]]:
        ix, seqs = ix_seqs
        (extreme,) = frequency_extremes(map(itemgetter(ix), seqs), reducer)
        return ix + 1, list(filter(lambda s: s[ix] == extreme, seqs))

    filtered = iterate(filter_, (0, seqs))
    final = next(filter(lambda ix_seqs: len(ix_seqs[1]) == 1, filtered))[1]
    return final[0]


def to_int(binary: Sequence[bool]) -> int:
    places = map((2).__pow__, range(len(binary)))
    return sum(map(mul, reversed(binary), places))


def _min_or_max_by_count(
    reducer: Callable[[Iterable[Tuple[int, T]]], Tuple[int, T]],
    counts: Iterable[Tuple[T, int]],
) -> T:
    # bias towards lower key in case of a count tie
    return tuple(reversed(reducer(map(tuple, map(reversed, counts)))))


min_by_count = partial(_min_or_max_by_count, min)
max_by_count = partial(_min_or_max_by_count, max)


if __name__ == "__main__":
    import sys

    with open("day03.txt") if sys.stdin.isatty() else sys.stdin as f:
        lines = filter(bool, map(str.strip, f))
        seqs = list(map(list, map(partial(map, int), lines)))
        n_digits = len(seqs[0])

    # Part 1

    digits_gamma, digits_epsilon = zip(
        *map(least_and_most_common, (map(itemgetter(i), seqs) for i in range(n_digits)))
    )
    print(to_int(digits_gamma) * to_int(digits_epsilon))

    # Part 2

    digits_O2 = filter_by_frequency(seqs, min_by_count)
    digits_CO2 = filter_by_frequency(seqs, max_by_count)
    print(to_int(digits_O2) * to_int(digits_CO2))
