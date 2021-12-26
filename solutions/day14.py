from collections import defaultdict, Counter
from functools import partial
from itertools import accumulate, chain, islice, repeat, tee
from typing import (
    Callable,
    Dict,
    Hashable,
    Iterable,
    Iterator,
    List,
    Mapping,
    Tuple,
    TypeVar,
)

T = TypeVar("T", bound=Hashable)
Pair = Tuple[T, T]
InsertionLookup = Dict[Pair[T], T]
SymbolCounts = Mapping[T, int]
PairCounts = Mapping[Pair[T], int]

START, END = object(), object()


def substitute_counts(
    lookup: InsertionLookup[T], counts: PairCounts[T]
) -> PairCounts[T]:
    new_counts = defaultdict(int)
    for (a, b), count in counts.items():
        c = lookup.get((a, b))
        if c is None:
            new_counts[(a, b)] += count
        else:
            new_counts[(a, c)] += count
            new_counts[(c, b)] += count
    return new_counts


def substitute_many_counts(
    n_steps: int, lookup: InsertionLookup[T], seq: Iterable[T]
) -> PairCounts[T]:
    sub = partial(substitute_counts, lookup)
    return iterate_n(sub, n_steps, Counter(pairs(seq)))


def iterate_n(f: Callable[[T], T], n: int, x: T) -> T:
    it = iterate(f, x)
    tail = islice(it, n, None)
    return next(tail)


def iterate(f: Callable[[T], T], x: T) -> Iterator[T]:
    def call(x, f):
        return f(x)

    return accumulate(repeat(f), call, initial=x)


def pairs(seq: Iterable[T]) -> Iterator[Pair[T]]:
    l, r = tee(chain((START,), seq, (END,)), 2)
    next(r)
    return zip(l, r)


def to_symbol_counts(pair_counts: PairCounts[T]) -> SymbolCounts[T]:
    counts = defaultdict(int)
    for (a, b), n in pair_counts.items():
        if a is not START:
            counts[a] += n
        if b is not END:
            counts[b] += n
    # every symbol adds 2 to the count because of the START, END placeholders
    return {k: (v + 1) // 2 for k, v in counts.items()}


def pair_counts(seq: Iterable[T]) -> PairCounts[T]:
    return Counter(pairs(seq))


# Input parsing


def parse_input(f) -> Tuple[List[str], InsertionLookup[str]]:
    rules = {}
    seq = list(f.readline().strip())
    for line in filter(None, map(str.strip, f)):
        pair, sub = line.split(" -> ")
        rules[tuple(pair)] = sub
    return seq, rules


def test():
    print("running tests")
    import io

    f = io.StringIO(
        """NNCB

    CH -> B
    HH -> N
    CB -> H
    NH -> C
    HB -> C
    HC -> B
    HN -> C
    NN -> C
    BH -> H
    NC -> B
    NB -> B
    BN -> B
    BB -> N
    BC -> B
    CC -> N
    CN -> C"""
    )
    seq, rules = parse_input(f)
    final_seq = "NBBNBNBBCCNBCNCCNBBNBBNBBBNBBNBBCBHCBHHNHCBBCBHCB"

    assert (substitute_many_counts(4, rules, seq)) == Counter(pairs(final_seq))
    final_counts = to_symbol_counts(substitute_many_counts(4, rules, seq))
    assert final_counts == Counter(final_seq)
    from random import choice

    def random_seq(n):
        return "".join(choice("ABCDEFGHIJKLMNOP") for i in range(n))

    for i in range(1000):
        s = random_seq(100)
        assert to_symbol_counts(pair_counts(s)) == Counter(s), s


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test()

    with open("day14.txt") if sys.stdin.isatty() else sys.stdin as f:
        seq, rules = parse_input(f)

    def answer(n: int):
        pair_counts = substitute_many_counts(n, rules, seq)
        symbol_counts = to_symbol_counts(pair_counts)
        return max(symbol_counts.values()) - min(symbol_counts.values())

    # Part 1

    print(answer(10))

    # Part 2

    print(answer(40))
