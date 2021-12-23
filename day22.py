from collections import defaultdict, deque
from functools import reduce
from itertools import product
from operator import mul
import re
from typing import (
    Callable,
    Dict,
    Hashable,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
)

T = TypeVar("T", bound=Hashable)
D = TypeVar("D")
Label = TypeVar("Label", bound=Hashable)
Coord = Tuple[int, int, int]
Range = Tuple[int, int]
Cuboid = Tuple[Range, ...]
Instruction = Tuple[Cuboid, Label]
IndexedInstruction = Tuple[int, Instruction]
DataGraph = Dict[T, Dict[T, D]]


# Main Function


def final_state(instructions: Iterable[Instruction[Label]]) -> Dict[Label, int]:
    label_counts = defaultdict(int)
    graph: DataGraph[IndexedInstruction[Label], Cuboid] = dict()
    for step, (cuboid, label) in enumerate(instructions):
        neighbors = {
            (step2, (cuboid2, label2)): cuboid_intersection(cuboid, cuboid2)
            for step2, (cuboid2, label2) in graph
            if cuboid_intersection(cuboid, cuboid2)
        }
        graph[(step, (cuboid, label))] = neighbors

        for path, intersection in all_paths(
            graph, ((step, (cuboid, label)), cuboid), reducer=cuboid_intersection,
        ):
            if intersection is not None:
                size = cuboid_volume(intersection)
                label2 = path[-1][1][1]
                # inclusion-exclusion
                increment = ((-1) ** len(path)) * size
                label_counts[label2] -= increment

    return label_counts


# Helpers


def cuboid_volume(cuboid: Optional[Cuboid]) -> int:
    return 0 if cuboid is None else reduce(mul, (hi + 1 - lo for lo, hi in cuboid))


def cuboid_intersection(
    cuboid1: Optional[Cuboid], cuboid2: Optional[Cuboid]
) -> Optional[Cuboid]:
    if cuboid1 is None or cuboid2 is None:
        return None
    result = tuple(
        (max(lo1, lo2), min(hi1, hi2))
        for (lo1, hi1), (lo2, hi2) in zip(cuboid1, cuboid2)
    )
    if any(lo > hi for lo, hi in result):
        return None
    return result


def all_paths(
    graph: DataGraph[T, D], node_data: Tuple[T, D], reducer: Callable[[D, D], D]
) -> Iterator[Tuple[List[T], D]]:
    node, data = node_data
    paths = deque([([node], data)])
    while paths:
        path, data = paths.popleft()
        yield path, data
        neighbors = graph.get(path[-1])
        if neighbors:
            paths.extend(
                (path + [node], reducer(data, d)) for node, d in neighbors.items()
            )


# Input parsing


def parse_instruction(s: str) -> Instruction:
    i = r"-?\d+"
    fields = re.fullmatch(
        fr"(on|off)\s+x=({i})..({i}),\s*y=({i})..({i}),\s*z=({i})..({i})",
        s,
        re.IGNORECASE,
    ).groups()
    on = {"on": True, "off": False}[fields[0].lower()]
    return (
        (
            (int(fields[1]), int(fields[2])),
            (int(fields[3]), int(fields[4])),
            (int(fields[5]), int(fields[6])),
        ),
        on,
    )


# Tests


def test():
    # Naive implementation

    class ReactorState:
        def __init__(self, instructions: Iterable[Instruction[Label]], dimension: int):
            self.instructions = list(instructions)
            self.cuboid: Cuboid = tuple(
                (
                    min(c[i][0] for c, _ in self.instructions),
                    max(c[i][1] for c, _ in self.instructions),
                )
                for i in range(dimension)
            )

        def state_of(self, coord: Coord) -> Optional[Label]:
            instructions = filter(
                lambda ins: contains_coord(ins[0], coord), reversed(self.instructions)
            )
            last_cuboid, label = next(instructions, (None, None))
            return label

        def counts(self) -> Dict[Label, int]:
            counts = defaultdict(int)
            for coord in coords_in(self.cuboid):
                counts[self.state_of(coord)] += 1
            return counts

    def contains_coord(cuboid: Cuboid, coord: Coord) -> bool:
        return all(lo <= x and hi >= x for (lo, hi), x in zip(cuboid, coord))

    def coords_in(cuboid: Cuboid) -> Iterator[Coord]:
        ranges = (range(lo, hi + 1) for lo, hi in cuboid)
        return product(*ranges)

    from itertools import chain, combinations
    from operator import itemgetter
    import random
    import time

    def random_range(max_start: int, min_size: int, max_size: int) -> Range:
        lo = random.randint(0, max_start)
        size = random.randint(min_size, max_size)
        return (lo, lo + size)

    def random_cuboid(
        dimension: int, max_start: int, min_size: int, max_size: int
    ) -> Cuboid:
        return tuple(
            random_range(max_start, min_size, max_size) for _ in range(dimension)
        )

    def random_instructions(
        dimension: int,
        n_cuboids: int,
        states: List[Label],
        max_start: int = 20,
        min_size: int = 5,
        max_size: int = 20,
    ) -> List[Instruction[Label]]:
        return [
            (
                random_cuboid(dimension, max_start, min_size, max_size),
                random.choice(states),
            )
            for _ in range(n_cuboids)
        ]

    print("running tests")

    bools = [True, False]
    strings = ["a", "b", "c"]
    for dimension, n_cuboids, labels in product([2, 3], [5, 10, 20], [bools, strings]):
        instructions = random_instructions(dimension, n_cuboids, labels)
        overlaps2 = sum(
            1
            for (c1, _), (c2, _) in combinations(instructions, 2)
            if cuboid_intersection(c1, c2)
        )
        overlaps3 = sum(
            1
            for (c1, _), (c2, _), (c3, _) in combinations(instructions, 3)
            if reduce(cuboid_intersection, (c1, c2, c3))
        )
        lo = min(map(itemgetter(0), chain.from_iterable(c for c, _ in instructions)))
        hi = max(map(itemgetter(0), chain.from_iterable(c for c, _ in instructions)))
        print(
            f"{n_cuboids} cuboids with {len(labels)} labels in {dimension} dimensions, "
            f"coordinate ranges {lo}-{hi}, "
            f"{overlaps2} pairwise overlaps, {overlaps3} 3-way overlaps"
        )
        tic = time.time()
        efficient = final_state(instructions)
        efficient_time = time.time() - tic
        print(f"Efficient solution: {efficient_time:2.3f}")

        tic = time.time()
        naive = ReactorState(instructions, dimension).counts()
        naive_time = time.time() - tic
        print(f"Naive solution: {naive_time:2.3f}")
        print(f"Speedup: {naive_time / efficient_time:3.2f}")

        for label in labels:
            assert efficient[label] == naive[label]


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test()

    with open("day22.txt") if sys.stdin.isatty() else sys.stdin as f:
        instructions = list(map(parse_instruction, filter(bool, map(str.strip, f))))

    # Part 1

    from functools import partial

    def instruction_in_cuboid(cuboid: Cuboid, instruction: Instruction) -> bool:
        cuboid2, _ = instruction
        return all(
            lo1 <= lo2 and hi1 >= hi2 for (lo1, hi1), (lo2, hi2) in zip(cuboid, cuboid2)
        )

    ranges = ((-50, 50),) * 3
    result = final_state(filter(partial(instruction_in_cuboid, ranges), instructions))
    print(dict(result))

    # Part2

    result = final_state(instructions)
    print(dict(result))
