from itertools import accumulate, repeat
from typing import Callable, Iterable, Tuple, TypeVar
import numpy as np

T = TypeVar("T")
Grid = np.ndarray
Coords = Tuple[np.ndarray, np.ndarray]
State = Tuple[Grid, Coords, Coords, bool]

EMPTY = "."
SOUTHER = "v"
EASTER = ">"


def step(state: State) -> State:
    grid, (easter_x, easter_y), (souther_x, souther_y), _ = state
    new_grid = np.copy(grid)

    easter_x_new = (easter_x + 1) % grid.shape[1]
    discard_easters = new_grid[easter_y, easter_x_new] == EMPTY
    changed_easters = np.any(discard_easters)
    keep_easters = ~discard_easters
    easter_x_new[keep_easters] = easter_x[keep_easters]
    if changed_easters:
        new_grid[easter_y[discard_easters], easter_x[discard_easters]] = EMPTY
        new_grid[easter_y[discard_easters], easter_x_new[discard_easters]] = EASTER

    souther_y_new = (souther_y + 1) % grid.shape[0]
    discard_southers = new_grid[souther_y_new, souther_x] == EMPTY
    changed_southers = np.any(discard_southers)
    keep_southers = ~discard_southers
    souther_y_new[keep_southers] = souther_y[keep_southers]
    if changed_southers:
        new_grid[souther_y[discard_southers], souther_x[discard_southers]] = EMPTY
        new_grid[souther_y_new[discard_southers], souther_x[discard_southers]] = SOUTHER

    return (
        new_grid,
        (easter_x_new, easter_y),
        (souther_x, souther_y_new),
        changed_easters or changed_southers,
    )


def iterate(f: Callable[[T], T], x: T) -> Iterable[T]:
    return accumulate(repeat(f), lambda x, f: f(x), initial=x)


def simulate(grid: Grid) -> Tuple[Grid, int]:
    easters = np.where(grid == EASTER)[::-1]
    southers = np.where(grid == SOUTHER)[::-1]
    states = iterate(step, (grid, easters, southers, True))
    for i, (grid, _, _, changed) in enumerate(states):
        if not changed:
            return grid, i


def parse_array(f: Iterable[str]) -> Grid:
    return np.array(list(map(list, takewhile("".__ne__, map(str.rstrip, f)))))


if __name__ == "__main__":
    from itertools import takewhile
    import sys

    if len(sys.argv) > 1 and "-v" in sys.argv or "--verbose" in sys.argv:
        VERBOSE = True

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test()

    with open("day25.txt") if sys.stdin.isatty() else sys.stdin as f:
        grid = parse_array(f)

    # Part 1

    final_grid, n_steps = simulate(grid)
    print(n_steps)
