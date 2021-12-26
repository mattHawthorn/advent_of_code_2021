from functools import reduce
from typing import Iterable, Tuple
import numpy as np

Grid = np.ndarray
Fold = Tuple[int, int]


# Main function


def fold_many(grid: Grid, folds: Iterable[Fold]):
    return reduce(fold, folds, grid)


def fold(grid: Grid, fold_instruction: Fold) -> Grid:
    axis, index = fold_instruction
    if axis == 1:
        return fold(grid.T, (0, index)).T
    elif axis == 0:
        size1, size2 = index, grid.shape[0] - index - 1
        new_size = max(size1, size2)
        new = np.zeros((new_size, grid.shape[1]), dtype=bool)
        new[new_size - size1 :] |= grid[:size1]
        new[new_size - size2 :] |= grid[-1 : -1 - size2 : -1]
        return new
    else:
        raise ValueError(axis)


# Input parsing


def parse_point(line: str):
    x, y = map(int, line.strip().split(","))
    return x, y


def parse_fold(line: str):
    prefix = "fold along "
    suffix = line.strip()[len(prefix) :]
    axis, value = suffix.split("=")
    return ["y", "x"].index(axis), int(value)


def parse_input(f):
    xs, ys = [], []
    folds = []
    for line in f:
        if not line.strip():
            break
        else:
            x, y = parse_point(line)
            xs.append(x), ys.append(y)
    for line in f:
        folds.append(parse_fold(line))

    grid = np.zeros((max(ys) + 1, max(xs) + 1), dtype=bool)
    grid[ys, xs] = True
    return grid, folds


def test():
    print("running tests")
    import io

    grid, folds = parse_input(
        io.StringIO(
            """6,10
    0,14
    9,10
    0,3
    10,4
    4,11
    6,0
    6,12
    4,1
    0,13
    10,12
    3,4
    3,0
    8,4
    1,10
    2,14
    8,10
    9,0

    fold along y=7
    fold along x=5"""
        )
    )

    assert (
        fold_many(grid, folds)
        == np.array(
            [
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
                [False, False, False, False, False],
                [False, False, False, False, False],
            ]
        )
    ).all()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test()

    with open("day13.txt") if sys.stdin.isatty() else sys.stdin as f:
        grid, folds = parse_input(f)

    # Part 1

    final_grid = fold_many(grid, folds[:1])
    print(final_grid.sum())

    # Part 2

    final_grid = fold_many(grid, folds)
    print(
        "\n".join(
            "".join(map({False: " ", True: "#"}.__getitem__, row)) for row in final_grid
        )
    )
