from functools import reduce
from itertools import takewhile
from typing import Callable, Iterator, Tuple, TypeVar

import numpy as np

FLASH_THRESHOLD = 9
RESET_VALUE = 0

T = TypeVar("T")
Grid = np.ndarray


def step(grid: Grid) -> Tuple[Grid, int]:
    return flash_cascade(grid + 1)


def flash_cascade(grid: Grid) -> Tuple[Grid, int]:
    unflashed = np.ones(grid.shape, dtype=bool)
    initial = (grid.copy(), unflashed, 0)
    states = takewhile(lambda t: t[2] > 0, iterate(_flash, initial))
    final_grid, unflashed, total_flashes = reduce(
        lambda t1, t2: (t2[0], t2[1], t1[2] + t2[2]), states, initial,
    )
    return final_grid, total_flashes


def iterate(f: Callable[[T], T], x: T) -> Iterator:
    while True:
        x = f(x)
        yield x


def _flash(state: Tuple[Grid, Grid, int]) -> Tuple[Grid, Grid, int]:
    grid, unflashed, num_flashes = state
    # mutates grid; pass a copy to avoid errors
    flashes = (grid > FLASH_THRESHOLD) & unflashed
    x_pos, y_pos = np.where(flashes)
    if len(x_pos) > 0:
        # reset coords flashed in this stage
        grid[x_pos, y_pos] = RESET_VALUE
        unflashed[x_pos, y_pos] = False
        # increment flashed coordinates' neighbors
        for x_inc, y_inc in [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]:
            new_x_pos = x_pos + x_inc
            new_y_pos = y_pos + y_inc
            x_in_bounds = (new_x_pos >= 0) & (new_x_pos < grid.shape[0])
            y_in_bounds = (new_y_pos >= 0) & (new_y_pos < grid.shape[1])
            in_bounds = x_in_bounds & y_in_bounds
            new_x_pos, new_y_pos = new_x_pos[in_bounds], new_y_pos[in_bounds]
            not_flashed = unflashed[new_x_pos, new_y_pos]
            new_x_pos, new_y_pos = new_x_pos[not_flashed], new_y_pos[not_flashed]
            grid[new_x_pos, new_y_pos] += 1

    return grid, unflashed, len(x_pos)


# Input parsing


def parse_array(s: str) -> np.ndarray:
    lines = s.strip().splitlines()
    values = [list(map(int, line.strip())) for line in lines]
    return np.array(values)


# Fun GIF output


def write_simulation(
    grid,
    filename: str = "day11.gif",
    size: int = 200,
    frame_duration_ms: int = 100,
    n_frames: int = 555,
):
    from itertools import chain, islice
    from PIL import Image

    def to_im(a, max_val=FLASH_THRESHOLD, size=(size, size)):
        return Image.fromarray((a * (255 / max_val)).astype("uint8")).resize(
            size, Image.BOX
        )

    def to_gif(ims, n_frames, filename, duration_ms):
        ims = list(islice(ims, n_frames))
        ims[0].save(
            filename,
            format="GIF",
            append_images=ims[1:],
            save_all=True,
            duration=duration_ms,
            loop=0,
        )

    states = chain((grid,), iterate(lambda g: step(g)[0], grid))
    ims = map(to_im, states)
    print(
        f"Writing {frame_duration_ms * n_frames / 1000:3.2f}s GIF with {n_frames:d} frames to {filename}"
    )
    to_gif(ims, n_frames, filename, frame_duration_ms)


def test():
    print("running tests")
    a = parse_array(
        """5483143223
    2745854711
    5264556173
    6141336146
    6357385478
    4167524645
    2176841721
    6882881134
    4846848554
    5283751526"""
    )
    a1 = parse_array(
        """6594254334
    3856965822
    6375667284
    7252447257
    7468496589
    5278635756
    3287952832
    7993992245
    5957959665
    6394862637"""
    )
    a2 = parse_array(
        """8807476555
    5089087054
    8597889608
    8485769600
    8700908800
    6600088989
    6800005943
    0000007456
    9000000876
    8700006848"""
    )

    assert (step(a)[0] == a1).all()
    assert (step(a1)[0] == a2).all()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test()
    if "--gif" in sys.argv:
        import json

        i = sys.argv.index("--gif")
        filename = sys.argv[i + 1] if len(sys.argv) > i + 1 else "day11.gif"
        options = json.loads(sys.argv[i + 2]) if len(sys.argv) > i + 2 else {}
        gif = True
    else:
        gif = False
        filename = None
        options = None

    with open("day11.txt") if sys.stdin.isatty() else sys.stdin as f:
        grid = parse_array(f.read())

    if gif:
        write_simulation(grid, filename, **options)

    # Part 1

    def step_accumulate(grid_num_flashes: Tuple[Grid, int]) -> Tuple[Grid, int]:
        grid, num_flashes = grid_num_flashes
        new_grid, new_num_flashes = step(grid)
        return new_grid, num_flashes + new_num_flashes

    from itertools import islice

    states = iterate(step_accumulate, (grid, 0))
    final_grid, num_flashes = next(islice(states, 100 - 1, None))
    print(num_flashes)

    # Part 2

    def step_(grid_num_flashes: Tuple[Grid, int]) -> Tuple[Grid, int]:
        grid, _ = grid_num_flashes
        return step(grid)

    indexed_states = enumerate(iterate(step_, (grid, 0)), 1)
    all_flashed = lambda t: t[1][1] == grid.size
    final_ix, (_, final_num_flashes) = next(filter(all_flashed, indexed_states))
    print(final_ix)
