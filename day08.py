from collections import ChainMap, Counter, defaultdict
from functools import partial, reduce
from itertools import chain, permutations
from math import factorial
from typing import (
    List,
    Hashable,
    FrozenSet,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
)

# Problem:
#
# Given:
# Universe U of items (e.g. the 7 segments)
# Set of subsets of U: S (e.g. the sets of segments used by each digit)
# Set of subsets of U: Sₚ (e.g. the sets of segments used by each digit in the scrambled output)
# (Note: Sₚ may be smaller than S)
#
# Find the permutation(s) p of U such that ∀ s ∈ Sₚ: p(s) ∈ S
#
# The function `solve` below solves this problem, iterating over all viable solutions, using a
# search path on candidate sets to map that is heuristically likely to be efficient
# (Note that p need not be a permutation of U; it may also be a bijection into another space W.
# The solution works in this more general setting as well, but we used one type variable T throughout
# for simplicity)

T = TypeVar("T", bound=Hashable)
Subset = FrozenSet[T]
SubsetsByLen = Mapping[int, Set[Subset]]
Solution = Mapping[T, T]
Input = Set[Subset]
Output = Sequence[Subset]


# Helpers


def solve(sets: Input, observed_sets: Input) -> Iterator[Solution]:
    sets_by_len = group_by_len(sets)
    search_path = efficient_path(observed_sets, sets_by_len)
    return solve_from_search_path(search_path, sets, observed_sets, sets_by_len)


def solve_from_search_path(
    search_path: Sequence[Subset],
    sets: Input,
    observed_sets: Input,
    sets_by_len: Optional[SubsetsByLen] = None,
) -> Iterator[Solution]:
    # every set on the search path is compatibly mapped by construction, but we need to check the rest
    compatible = partial(is_compatible, sets, observed_sets)
    if sets_by_len is None:
        sets_by_len = group_by_len(observed_sets)
    candidate_solutions = map(dict, candidate_mappings(search_path, sets_by_len))
    return filter(compatible, candidate_solutions)


def is_compatible(
    check_sets: Input, observed_sets: Iterable[Subset], mapping: Solution
):
    return all(map(check_sets.__contains__, translate_sets(observed_sets, mapping)))


def translate_sets(sets: Iterable[Subset], mapping: Solution) -> Iterator[Subset]:
    translate = partial(map, mapping.__getitem__)
    return map(frozenset, map(translate, sets))


def group_by_len(sets: Iterable[Subset]) -> SubsetsByLen:
    mapping = defaultdict(set)
    for s in sets:
        mapping[len(s)].add(s)
    return mapping


def candidate_mappings(
    search_path: Iterable[Subset], sets_by_len: SubsetsByLen
) -> Iterator[Solution]:
    def search(candidate_mappings, observed_set):
        # search of all possible mappings extending the current one by way of completing
        # the mapping to a particular observed set
        candidate_sets = sets_by_len[len(observed_set)]

        def extend_search(candidate_mapping):
            return chain.from_iterable(
                extended_candidate_mappings(og_set, observed_set, candidate_mapping)
                for og_set in candidate_sets
            )

        return chain.from_iterable(map(extend_search, candidate_mappings))

    return reduce(search, search_path, [{}])


def extended_candidate_mappings(
    og_set: Subset, observed_set: Subset, base_mapping: Mapping[str, str],
) -> Iterator[Solution]:
    # extend a current partial mapping using the possible permutations implied by og_set, observed_set.
    # compatibility check: everything mapped so far from the original set
    # must not map to something outside the target set
    assert len(og_set) == len(
        observed_set
    ), f"lengths must match; got {len(og_set)}, {len(observed_set)}"
    # unmapped symbols
    unmapped = observed_set.difference(base_mapping.keys())
    # mappable targets
    targets = og_set.difference(base_mapping.values())
    # can only extend the mapping using this pair of sets if doing so would still result in a bijection
    if len(unmapped) == len(targets):
        if (
            len(unmapped) == 0
        ):  # no new elements to map, but the current mapping checks out
            yield base_mapping
        for perm in permutations(targets):
            yield ChainMap(base_mapping, dict(zip(unmapped, perm)))


def efficient_path(
    sets: Set[Subset], observed_sets_by_len: Optional[Mapping[int, Set[Subset]]] = None
) -> List[Subset]:
    # find a sequence of sets, starting from the empty set, such that the cost of extending a
    # permutation search by the new members of each new set is minimized, and all members of all sets
    # are covered by at least one set in the sequence.
    # This _could_ be a single shortest path algorithm, if we had a cost function that was purely a
    # function of two subsets on an edge, e.g. the size of their difference. However, the change to the
    # cost function of a path for adding a new set to a path is actually a function of the _whole_ path
    # and the new set - namely a function of the size of the difference of the new set and the union of
    # all earlier sets on the path.
    #
    # To find the global optimum would require all-pairs shortest paths, filtering to paths that
    # cover all set members, and then choosing the shortest (least cost) one.
    # For simplicity we choose a greedy algorithm here which is worst case O(V^2)
    # (where V is the number of sets in this case)
    sets = sets.copy()
    order = []
    covered = set()
    universe_size = len(set(chain.from_iterable(sets)))

    def cost_fn(s):
        # for each observed set of the same length, we have to try to map the new elements
        # to the unmapped elements of U
        new_elts = len(s.difference(covered))
        c = factorial(new_elts)
        return (
            c * len(observed_sets_by_len[len(s)])
            if observed_sets_by_len is not None
            else c
        )

    while len(covered) < universe_size:
        smallest_diff = min(sets, key=cost_fn)
        sets.remove(smallest_diff)
        covered.update(smallest_diff)
        order.append(smallest_diff)

    return order


# Input parsing


def parse_line(line: str) -> Tuple[Input, Output]:
    i, o = line.strip().split("|")
    return (
        set(map(frozenset, i.strip().split())),
        list(map(frozenset, o.strip().split())),
    )


# Tests/benchmarking


def test():
    print("running tests")
    for universe_size, num_sets, min_size, max_size in [
        (10, 20, 2, 10),
        (15, 30, 2, 15),
        (20, 40, 2, 20),
    ]:
        print(
            f"Universe size: {universe_size}, num sets: {num_sets}, "
            f"min size: {min_size}, max size: {max_size}"
        )
        test_one(universe_size, num_sets, min_size, max_size, 0.8)


def test_one(
    universe_size: int,
    num_sets: int,
    min_set_size: int,
    max_set_size: int,
    subset_proportion: float,
):
    from random import choice, random, sample
    import time

    # Generate universe U (characters
    universe = list(map(chr, range(universe_size)))
    # Generate subsets S
    sizes = range(min_set_size, max_set_size + 1)
    subsets = {frozenset(sample(universe, choice(sizes))) for _ in range(num_sets)}
    print(f"Sizes: {dict(sorted(Counter(map(len, subsets)).items()))}")
    # universe actually sampled from
    universe = set(chain.from_iterable(subsets))
    # create mapping from U to int index in random order (generate permutation P)
    known_inverse_mapping = dict(enumerate(sample(universe, len(universe))))
    mapping = dict(map(reversed, known_inverse_mapping.items()))
    mapped_subsets = list(translate_sets(subsets, mapping))
    # translate original sets to new space of ints
    while True:
        # take a proper subset of the original subsets
        observed_sets = set(s for s in mapped_subsets if random() < subset_proportion)
        if len(set(chain.from_iterable(observed_sets))) == len(universe):
            break
    print(
        f"Observed set sizes ({len(observed_sets)} out of {len(subsets)}): "
        f"{dict(sorted(Counter(map(len, subsets)).items()))}"
    )
    # check that original mapping recovers original sets
    assert set(translate_sets(observed_sets, known_inverse_mapping)).issubset(subsets)
    # property-based test: translation of scrambled sets by solution mapping should equal original sets
    tic = time.time()
    solutions = solve(subsets, observed_sets)
    other_inverse_mapping = next(solutions)
    toc = time.time()
    print(f"{toc - tic:2.3f}s runtime")
    assert set(translate_sets(observed_sets, other_inverse_mapping)).issubset(subsets)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test()
        exit(0)

    with open("day08.txt") if sys.stdin.isatty() else sys.stdin as f:
        lines = list(map(parse_line, f))

    # Constants specific to the Advent of Code problem

    sets = list(
        map(
            frozenset,
            [
                "aefgbc",
                "cf",
                "aegdc",
                "afgdc",
                "cdbf",
                "afgbd",
                "aefgbd",
                "caf",
                "aefgbdc",
                "afgbdc",
            ],
        )
    )
    set_to_int = dict(map(reversed, enumerate(sets)))
    sets = set(sets)

    def decode(input_, output):
        solution = next(solve(sets, input_))
        return list(map(set_to_int.__getitem__, translate_sets(output, solution)))

    # Part 1

    from itertools import starmap

    digits = starmap(decode, lines)
    counts = Counter(chain.from_iterable(digits))
    print(sum(map(counts.__getitem__, [1, 4, 7, 8])))

    # Part 2

    digits = starmap(decode, lines)
    ints = (int("".join(map(str, _))) for _ in digits)
    print(sum(ints))
