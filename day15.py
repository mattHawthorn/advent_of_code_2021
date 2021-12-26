from collections import defaultdict
from functools import partial
from heapq import heappop, heappush
from itertools import repeat
from operator import add, itemgetter
from typing import Hashable, List, Set, Mapping, Tuple, TypeVar

import numpy as np

Grid = np.ndarray
T = TypeVar("T", bound=Hashable)
WeightedGraph = Mapping[T, Set[Tuple[T, int]]]


def grid_to_graph(grid: Grid) -> WeightedGraph[T]:
    graph = defaultdict(list)
    all_, head, tail = slice(None, None), slice(None, -1), slice(1, None)
    l, r, u, d = (all_, head), (all_, tail), (head, all_), (tail, all_)
    n_row, n_col = grid.shape
    xs = np.repeat(range(n_row), n_col).reshape(n_row, n_col)
    ys = np.tile(range(n_col), n_row).reshape(n_row, n_col)

    def add_weighted_edges(graph, edges):
        for a, b, w in edges:
            graph[a].append((b, w))

    for ixs1, ixs2 in (l, r), (r, l), (u, d), (d, u):
        add_weighted_edges(
            graph,
            zip(
                zip(xs[ixs1].flat, ys[ixs1].flat),
                zip(xs[ixs2].flat, ys[ixs2].flat),
                grid[ixs2].flat,
            ),
        )

    return graph


def djikstra(graph: WeightedGraph[T], start: T, end: T) -> Tuple[List[T], int]:
    unvisited = set(graph)
    dists = dict(zip(graph, repeat(float("inf"))))
    dists[start] = 0
    heap = []
    preds = {}

    def update_dists(node_dists, node):
        for n, dist in node_dists:
            if dist < dists[n]:
                dists[n] = dist
                preds[n] = node
                heappush(heap, (dist, n))

    def explore(node):
        dist = dists[node]
        nbrs = [(n, dist) for n, dist in graph[node] if n in unvisited]
        nbr_dists = map(partial(add, dist), map(itemgetter(1), nbrs))
        new_dists = zip(map(itemgetter(0), nbrs), nbr_dists)
        update_dists(new_dists, node)
        unvisited.remove(node)
        return heappop(heap)[1]

    node = start
    while node != end:
        node = explore(node)

    path = []
    node = end
    while node != start:
        path.append(node)
        node = preds[node]
    path.append(node)

    return list(reversed(path)), dists[end]


# Input parsing

Coord = Tuple[int, int]


def parse_grid(f) -> Grid:
    return tile_grid(np.array([list(map(int, row.strip())) for row in f]))


def grid_to_input(grid: Grid) -> Tuple[WeightedGraph[Coord], Coord, Coord]:
    return grid_to_graph(grid), (0, 0), (grid.shape[0] - 1, grid.shape[1] - 1)


def tile_grid(grid):
    x, y = grid.shape
    new_grid = np.empty((x * 5, y * 5), dtype=int)
    for i in range(5):
        for j in range(5):
            new_grid[i * x : i * x + x, j * y : j * y + y] = incriment_grid(grid, i + j)
    return new_grid


def incriment_grid(grid, n):
    new = (grid + n) % 9
    new[new == 0] = 9
    return new


def test():
    import io

    print("running tests")

    f = io.StringIO(
        """1163751742
    1381373672
    2136511328
    3694931569
    7463417111
    1319128137
    1359912421
    3125421639
    1293138521
    2311944581"""
    )
    grid = parse_grid(f)[:10, :10]
    graph, start, end = grid_to_input(grid)
    path, dist = djikstra(graph, start, end)

    def print_path(path, grid):
        x = max(map(itemgetter(0), path))
        y = max(map(itemgetter(1), path))
        points = set(path)
        for i in range(x + 1):
            line = "".join(
                str(grid[i, j]) if (i, j) in points else " " for j in range(y + 1)
            )
            print(line)

    print_path(path, grid)
    assert dist == 40


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test()

    with open("day15.txt") if sys.stdin.isatty() else sys.stdin as f:
        grid = parse_grid(f)

    # Part 1

    graph, start, stop = grid_to_input(grid[: grid.shape[0] // 5, : grid.shape[1] // 5])
    path, dist = djikstra(graph, start, stop)
    print(dist)

    # Part 2

    graph, start, stop = grid_to_input(grid)
    _, dist = djikstra(graph, start, stop)
    print(dist)
