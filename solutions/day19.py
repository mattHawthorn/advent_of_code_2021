from collections import defaultdict, deque
from functools import partial
from itertools import chain, permutations, product, starmap
from operator import matmul
import re
from typing import (
    IO,
    Collection,
    Dict,
    Iterable,
    Iterator,
    Optional,
    Mapping,
    Set,
    Tuple,
)
import numpy as np

Coord = Tuple[int, int, int]
Rotation = np.ndarray
Translation = Tuple[Rotation, Coord]

OVERLAP_THRESHOLD = 12


# Vector ops


def permutation_matrix(permutation: Coord):
    # determines the destination of a unit cube face
    a = np.zeros((3, 3), dtype=int)
    a[permutation, range(3)] = 1
    # multiply by +/-1 in case there is a reflection involved
    return a * int(round(np.linalg.det(a)))


def rotation_matrix_xy(n_90: int) -> Rotation:
    # determines the orientation of a unit cube face
    cos, sin = [(1, 0), (0, 1), (-1, 0), (0, -1)][n_90]
    return np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1],], dtype=int,)


all_permutations = list(map(permutation_matrix, permutations((0, 1, 2))))
all_xy_rotations = list(map(rotation_matrix_xy, range(4)))
all_rotations = list(starmap(matmul, product(all_permutations, all_xy_rotations)))


def rotate(rotation: Rotation, coord: Coord) -> Coord:
    return tuple(rotation @ coord)


def translate(translation: Optional[Translation], coord: Coord) -> Coord:
    # rotate, then shift
    if translation is None:
        return coord
    rotation, shift = translation
    return add(shift, rotate(rotation, coord))


def compose(trans1: Translation, trans2: Translation) -> Translation:
    # result of applying trans1, then trans2
    # v2 + r2(v1 + r1(x)) -> v2 + r2(v1) + r2(r1(x))
    rotation1, shift1 = trans1
    rotation2, shift2 = trans2
    return rotation2 @ rotation1, add(shift2, rotate(rotation2, shift1))


def add(coord1: Coord, coord2: Coord) -> Coord:
    return tuple(np.array(coord2, copy=False) + np.array(coord1, copy=False))


def diff(coord1: Coord, coord2: Coord) -> Coord:
    return tuple(np.array(coord2, copy=False) - np.array(coord1, copy=False))


# Main function


def discombobulate(
    observations: Mapping[int, Set[Coord]], source_mappings: Mapping[int, Translation]
) -> Set[Coord]:
    translated = set(
        chain.from_iterable(
            apply_translation(translation, observations[i])
            for i, translation in source_mappings.items()
        )
    )
    return translated


def scanner_locations(source_mappings: Mapping[int, Translation]) -> Dict[int, Coord]:
    return {
        i: translate(translation, (0, 0, 0))
        for i, translation in source_mappings.items()
    }


def apply_translation(
    translation: Optional[Translation], coords: Iterable[Coord]
) -> Iterable[Coord]:
    return (
        coords if translation is None else map(partial(translate, translation), coords)
    )


def possible_translations(
    coords1: Collection[Coord], coords2: Collection[Coord]
) -> Iterator[Translation]:
    for r in all_rotations:
        diffcounter: Mapping[Coord, int] = defaultdict(int)
        candidate_diffs: Set[Coord] = set()
        for c1, c2 in product(coords1, map(partial(rotate, r), coords2)):
            # to go from c2 to c1, rotate by r and add d
            d = diff(c2, c1)
            if d not in candidate_diffs:
                diffcounter[d] += 1
                if diffcounter[d] >= OVERLAP_THRESHOLD:
                    candidate_diffs.add(d)
                    yield r, d


def recover_mappings(
    observations: Mapping[int, Set[Coord]], reference_id: int = 0
) -> Dict[int, Translation]:
    observations: Dict[int, Set[Coord]] = dict(
        observations.items()
    )  # copy, since we mutate here
    # these hold the final translations that recover each set of coordinats to the reference frame of
    # the coordinates identified by `reference_id`
    source_translations: Dict[int, Translation] = {}
    # build spanning tree - `links` maps from leaf toward root
    searchq = deque([(reference_id, observations.pop(reference_id))])
    n_sets = len(observations)
    while observations and searchq:
        current_id, start_set = searchq.popleft()
        extend_search_from = []
        for linked_id, link_translation in find_translations(
            start_set, observations.items()
        ):
            # we keep track of these outside of `searchq` to avoid copying `observations` in the call to
            # `find_translations`, which would otherwise be required by deleting from it during iteration
            extend_search_from.append(linked_id)
            source_translation = source_translations.get(current_id)
            source_translations[linked_id] = (
                link_translation
                if source_translation is None
                else compose(link_translation, source_translation)
            )
        searchq.extend((i, observations.pop(i)) for i in extend_search_from)

    if len(source_translations) < n_sets:
        print(
            f"warning: only recovered mappings for {len(source_translations)} out of {n_sets} sets"
        )
    source_translations[reference_id] = None
    return source_translations


def find_translations(
    coords: Set[Coord], observations: Iterable[Tuple[int, Set[Coord]]]
) -> Iterator[Tuple[int, Translation]]:
    for i, others in observations:
        rot_trans = next(possible_translations(coords, others), None)
        if rot_trans is not None:
            rot, trans = rot_trans
            yield i, (rot, trans)


# Input parsing


def parse_input(f: IO) -> Mapping[int, Set[Coord]]:
    mapping = {}
    while True:
        i = parse_id(f)
        if i is None:
            break
        mapping[i] = parse_coords(f)
    return mapping


def parse_id(f: IO) -> Optional[int]:
    for line in map(str.strip, f):
        if line:
            scanner_id = int(re.search("scanner (\d+)", line).group(1))
            return scanner_id
    return None


def parse_coords(f: IO) -> Set[Coord]:
    lines = iter(lambda: f.readline().strip(), "")
    return set(map(parse_coord, lines))


def parse_coord(s: str) -> Coord:
    parts = tuple(map(int, re.fullmatch("(-?\d+),(-?\d+),(-?\d+)", s).groups()))
    return parts[0], parts[1], parts[2]


if __name__ == "__main__":
    import sys

    with open("day19.txt") if sys.stdin.isatty() else sys.stdin as f:
        id_to_coords = parse_input(f)

    # Part 1

    source_mappings = recover_mappings(id_to_coords)
    mapped = discombobulate(id_to_coords, source_mappings)
    print(len(mapped))

    locations = scanner_locations(source_mappings)
    manhattan = lambda v1, v2: sum(map(abs, diff(v1, v2)))
    print(len(locations))
    print(max(starmap(manhattan, permutations(locations.values(), 2))))
