from heapq import heappush, heappop
from itertools import chain, filterfalse
from operator import itemgetter
from typing import Callable, Collection, Dict, Iterable, Iterator, List, Set, Tuple

Grid = List[List[int]]
Coord = Tuple[int, int]
Nbrhood = List[Tuple[Coord, int]]
Graph = Dict[Coord, List[Coord]]

# Helpers

rectilinear = ((-1, 0), (1, 0), (0, -1), (0, 1))


def filter_points(
    pattern: Collection[Coord], grid: Grid, predicate: Callable[[int, Nbrhood], bool]
) -> Iterator[Tuple[Coord, int]]:
    for coord, value, nbrhood in neighborhoods(grid, pattern):
        if predicate(value, nbrhood):
            yield coord, value


def neighborhoods(
    grid: Grid, pattern: Collection[Coord],
) -> Iterator[Tuple[Coord, int, Nbrhood]]:
    n_y = len(grid)
    n_x = len(grid[0])
    for i, row in enumerate(grid):
        for j, value in enumerate(row):
            yield (i, j), value, lookup(pattern, i, j, grid, n_x, n_y)


def lookup(
    pattern: Collection[Coord], i: int, j: int, grid: Grid, n_x: int, n_y: int
) -> List[Tuple[Coord, int]]:
    values = []
    for i_, j_ in pattern:
        ii = i + i_
        jj = j + j_
        if 0 <= ii < n_y and 0 <= jj < n_x:
            values.append(((ii, jj), grid[ii][jj]))
    return values


def is_minimal(value: int, nbrhood: Nbrhood):
    return value < min(v for _, v in nbrhood)


def build_graph(
    grid: Grid,
    edge_predicate: Callable[[int, int], bool],
    pattern: Collection[Coord] = rectilinear,
) -> Tuple[Dict[Coord, int], Graph]:
    values = {}
    graph = {}
    for coord, value, nbrs in neighborhoods(grid, pattern):
        graph[coord] = [c for c, v in nbrs if edge_predicate(value, v)]
        values[coord] = value

    return values, graph


def connected_components(graph: Graph, seeds: Iterable[Coord]) -> Iterator[Set[Coord]]:
    components = map(partial(connected_component, graph), seeds)
    return components


def connected_component(graph: Graph, seed: Coord) -> Set[Coord]:
    component = set()
    shell = {seed}
    while shell:
        new_shell = set(
            chain.from_iterable(
                map(graph.__getitem__, filterfalse(component.__contains__, shell))
            )
        )
        component.update(shell)
        shell = new_shell

    return component


# Main functions


def low_points(grid: Grid, pattern=rectilinear) -> Iterator[Tuple[Coord, int]]:
    return filter_points(pattern, grid, is_minimal)


def components(
    grid: Grid, edge_predicate: Callable[[int, int], bool]
) -> Iterator[Set[Coord]]:
    # connected components in the graph seeded by low points with edges moving along increasing paths
    values, graph = build_graph(grid, edge_predicate=edge_predicate)
    seeds = set(graph).difference(chain.from_iterable(graph.values()))
    return connected_components(graph, seeds)


def top_k(it: Iterable, k: int, key=lambda x: x) -> List:
    q = []
    for i in it:
        if len(q) < k:
            heappush(q, (key(i), i))
        else:
            smallest = heappop(q)
            k_ = key(i)
            heappush(q, (k_, i) if k_ > smallest[0] else smallest)

    result = []
    while q:
        result.append(heappop(q)[1])

    return result


if __name__ == "__main__":
    import sys

    with open("day9.txt") if sys.stdin.isatty() else sys.stdin as f:
        grid = [list(map(int, line.strip())) for line in f]

    from operator import add
    from functools import partial

    # Part 1
    low_points_ = low_points(grid)
    values = map(itemgetter(1), low_points_)
    increment = partial(add, 1)
    print(sum(map(increment, values)))

    # Part 2
    basins = components(
        grid, edge_predicate=lambda here, there: here < there and there < 9
    )
    basin1, basin2, basin3 = top_k(basins, 3, key=len)
    print(len(basin1) * len(basin2) * len(basin3))
