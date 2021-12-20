from functools import reduce
from itertools import chain, product, repeat
from typing import IO, Iterable, Tuple

import numpy as np

Image = np.ndarray
Lookup = np.ndarray

SIZE = 3
HALF_SIZE = SIZE // 2
DIGIT_VALUES = (2 ** np.arange(SIZE * SIZE - 1, -1, -1)).reshape(SIZE, SIZE)


def enhance(lookup: Lookup, image: Image, fill: bool = False) -> Tuple[Image, bool]:
    ints, new_fill_ix = convolve(DIGIT_VALUES, image, fill)
    new_image = lookup[ints]
    new_fill = lookup[new_fill_ix]
    return new_image, new_fill


def enhance_n(lookup: Lookup, image: Image, n: int, fill: bool = False, trim: bool = False) -> Tuple[Image, bool]:
    def enhance_(im_fill):
        return enhance(lookup, *im_fill)

    new_image, new_fill = reduce(lambda a, f: f(a), repeat(enhance_, n), (image, fill))
    if trim:
        trim0, trim1 = ((new_image.shape[i] - image.shape[i]) // 2 for i in (0, 1))
        return new_image[trim0:-trim0, trim1:-trim1]
    else:
        return new_image


def convolve(neighborhood: np.ndarray, image: Image, fill: bool = False) -> Tuple[np.ndarray, int]:
    assert neighborhood.shape[0] == neighborhood.shape[1]
    size = neighborhood.shape[0]
    half_size = size // 2
    # pad out with `size - 1` rows/cols - these are the cells that are needed for the computation
    # everything outside this will be filled with the new fill value in an infinite grid
    padded_image = padded(image, size - 1, fill)
    # the convolved image is then smaller on all sides by `half_size`
    new_image = np.empty(
        (padded_image.shape[0] - 2 * half_size, padded_image.shape[1] - 2 * half_size), dtype=int
    )
    for i, j in product(range(new_image.shape[0]), range(new_image.shape[1])):
        swatch = padded_image[i:i+size, j:j+size]
        new_image[i, j] = (swatch * neighborhood).sum()

    new_fill = (neighborhood * np.full(neighborhood.shape, fill)).sum()
    return new_image, new_fill


def padded(image: np.ndarray, size: int, fill: bool = False) -> np.ndarray:
    padded_image = np.empty((image.shape[0] + 2 * size, image.shape[1] + 2 * size), dtype=image.dtype)
    for ix in range(size):
        padded_image[ix, :] = fill
        padded_image[-ix - 1, :] = fill
        padded_image[:, ix] = fill
        padded_image[:, -ix - 1] = fill

    padded_image[size:-size, size:-size] = image
    return padded_image


# Input Parsing

PIXEL_VALUES = {".": False, "#": True}


def parse_input(f: IO) -> Tuple[Lookup, Image]:
    lookup_rows = list(chain.from_iterable(map(to_binary, iter(lambda: f.readline().strip(), ""))))
    return np.array(lookup_rows, dtype=bool), parse_image(f)


def parse_image(lines: Iterable[str]) -> Image:
    image_rows = list(map(to_binary, filter(bool, map(str.strip, lines))))
    return np.array(image_rows, dtype=bool)


def to_binary(s: str):
    return list(map(PIXEL_VALUES.__getitem__, s))


def test():
    import io

    print("running tests")

    f = io.StringIO("""..#.#..#####.#.#.#.###.##.....###.##.#..###.####..#####..#....#..#..##..##
    #..######.###...####..#..#####..##..#.#####...##.#.#..#.##..#.#......#.###
    .######.###.####...#.##.##..#..#..#####.....#.#....###..#.##......#.....#.
    .#..#..##..#...##.######.####.####.#.#...#.......#..#.#.#...####.##.#.....
    .#..#...##.#.##..#...##.#.##..###.#......#.#.......#.#.#.####.###.##...#..
    ...####.#..#..#.##.#....##..#.####....##...##..#...#......#.#.......#.....
    ..##..####..#...#.#.#...##..#.#..###..#####........#..####......#..#
    
    #..#.
    #....
    ##..#
    ..#..
    ..###""")
    lookup, image = parse_input(f)
    expected = parse_image("""...............
    ...............
    ...............
    ...............
    .....##.##.....
    ....#..#.#.....
    ....##.#..#....
    ....####..#....
    .....#..##.....
    ......##..#....
    .......#.#.....
    ...............
    ...............
    ...............
    ...............""".splitlines())[4:-4, 4:-4]
    expected2 = parse_image("""...............
    ...............
    ...............
    ..........#....
    ....#..#.#.....
    ...#.#...###...
    ...#...##.#....
    ...#.....#.#...
    ....#.#####....
    .....#.#####...
    ......##.##....
    .......###.....
    ...............
    ...............
    ...............""".splitlines())[3:-3, 3:-3]
    def to_str(a):
        return "\n".join("".join(".#"[i] for i in row) for row in a)

    enhanced, _ = enhance(lookup, image)
    enhanced2 = enhance_n(lookup, image, 2, trim=False)
    assert (enhanced == expected).all(), f"{to_str(enhanced)}\n\n{to_str(expected)}"
    assert (enhanced2 == expected2).all(), f"{to_str(enhanced2)}\n\n{to_str(expected2)}"


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test()

    with open("day20.txt") if sys.stdin.isatty() else sys.stdin as f:
        lookup, image = parse_input(f)

    # Part 1

    enhanced = enhance_n(lookup, image, 2)
    print(np.sum(enhanced))

    # Part 2

    enhanced = enhance_n(lookup, image, 50)
    print(np.sum(enhanced))
