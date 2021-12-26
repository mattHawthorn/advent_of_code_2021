from collections import Counter
from heapq import heappop, heappush
from functools import partial
from itertools import accumulate, chain, islice, repeat, takewhile
from numbers import Number
from operator import is_, is_not
import re
from typing import (
    Callable,
    Dict,
    Hashable,
    Iterable,
    Iterator,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

T = TypeVar("T")
Node = TypeVar("Node", bound=Hashable)
Cost = TypeVar("Cost", bound=Number)
D = TypeVar("D")
Hops = Iterable[Tuple[Node, D]]
DataGraph = Mapping[Node, Hops[Node, D]]
QueueID = int

VERBOSE = False


class QueueNetwork(DataGraph[QueueID, Tuple[Sequence[D], Cost]]):
    def __init__(
        self,
        queues: Sequence[Sequence[D]],
        edges: Sequence[Sequence[Tuple[QueueID, Cost]]],
        cost_fn: Callable[[D], Cost],
        _safe=False,
    ):
        assert len(queues) == len(edges)
        # immutable
        self.queues = queues if _safe else tuple(map(tuple, queues))
        self.edges = edges if _safe else tuple(map(tuple, edges))
        self.cost_fn = cost_fn

    _hash: Optional[int] = None

    def __hash__(self) -> int:
        # to allow storing instances in sets or dicts; cache the result
        if self._hash is None:
            # cache the hash since it could be somewhat expensive to compute
            self._hash = hash(self.queues)
        return self._hash

    def __eq__(self, other: "QueueNetwork[D, Cost]"):
        return self.queues == other.queues

    def __repr__(self):
        return repr(self.queues)

    def __str__(self):
        return tostr(self)

    def moves(
        self,
    ) -> Iterable[Tuple["QueueNetwork[D, Cost]", D, QueueID, QueueID, Cost]]:
        # legal moves from one queue to another along edges, with cost adjusted for the queue length
        n_null = [sum(map(is_null, q)) for q in self.queues]
        full = [n == 0 for n in n_null]
        nonempty = [n < len(q) for q, n in zip(self.queues, n_null)]
        for q1, edges in enumerate(self.edges):
            for q2, cost in edges:
                if nonempty[q1] and not full[q2]:
                    qu1 = self.queues[q1]
                    qu2 = self.queues[q2]
                    move_ix, to_move = next(
                        (i, _) for i, _ in enumerate(reversed(qu1), 1) if not_null(_)
                    )
                    push_elts = list(takewhile(not_null, reversed(qu2)))
                    unit_cost = self.cost_fn(to_move)
                    total_cost = (cost + move_ix) * unit_cost + sum(
                        map(self.cost_fn, push_elts)
                    )
                    new_qu1 = (*qu1[:-move_ix], *repeat(None, abs(move_ix)))
                    new_qu2 = (*qu2[: -len(push_elts) - 1], *push_elts, to_move)
                    queues = list(self.queues)
                    queues[q1] = new_qu1
                    queues[q2] = new_qu2
                    new = QueueNetwork[D, Cost](
                        tuple(queues), self.edges, self.cost_fn, _safe=True
                    )
                    yield new, to_move, q1, q2, total_cost

    def __getitem__(self, queue_id: QueueID) -> Hops[QueueID, Tuple[Sequence[D], Cost]]:
        nodes_costs = self.edges[queue_id]
        return {n: (self.queues[n], cost) for n, cost in nodes_costs}

    def __iter__(self):
        return iter(range(len(self.queues)))

    def __len__(self):
        return len(self.queues)


class LazyDataGraph(DataGraph[Node, D]):
    def __init__(self, node_gen: Callable[[Node], Hops[Node, D]]):
        self.node_gen = node_gen

    def __getitem__(self, node: Node) -> Hops[Node, D]:
        return self.node_gen(node)

    def __len__(self):
        raise NotImplementedError(type(self))

    def __iter__(self):
        raise NotImplementedError(type(self))


class Heapable(NamedTuple):
    key: Cost
    value: Node

    def __lt__(self, other: "Heapable"):
        return self.key < other.key

    def __gt__(self, other: "Heapable"):
        return self.key > other.key


def djikstra(
    graph: DataGraph[Node, D],
    start: Node,
    is_end: Callable[[Node], bool],
    cost_fn: Callable[[D], Cost],
) -> Tuple[List[Node], Cost]:
    visited = set()
    dists: Dict[Node, Cost] = dict()
    dists[start] = 0
    heap: List[Tuple[Cost, Node]] = []
    predecessors = {}

    def explore(node: Node) -> Node:
        dist = dists[node]
        print_(f"explore from cost {dist} state:\n\n{node}", end="\n\n")

        n_visit = 0
        for n, data in graph[node]:
            if n not in visited:
                edge_dist = cost_fn(data)
                n_visit += 1
                prior_dist = dists.get(n)
                new_dist = dist + edge_dist
                if prior_dist is None or new_dist < prior_dist:

                    print_("new shortest path", new_dist)

                    dists[n] = new_dist
                    predecessors[n] = node
                    heappush(heap, Heapable(new_dist, n))

                print_()

        visited.add(node)

        try:
            _, next_node = heappop(heap)
        except IndexError:
            raise ValueError(
                f"No path exists from {start!r} to node satisfying {is_end!r}"
            )
        return next_node

    end = None
    for node in iterate(explore, start):
        if is_end(node):
            end = node
            break

    reverse_path = chain(
        takewhile(start.__ne__, iterate(predecessors.__getitem__, end)), (start,)
    )
    return list(reversed(list(reverse_path))), dists[end]


def iterate(f: Callable[[T], T], x: T) -> Iterator[T]:
    return accumulate(repeat(f), lambda x, f: f(x), initial=x)


def identity(x: T) -> T:
    return x


is_null = partial(is_, None)
not_null = partial(is_not, None)


def print_(*args, **kw):
    if VERBOSE:
        print(*args, **kw)


# Input parsing


def parse_input(f: Iterable[str]) -> Tuple[QueueNetwork, Dict[str, int]]:
    lines = enumerate(map(str.rstrip, f))
    hallway = ""
    hallway_bottom = ""
    for pattern in [r"#+", r"#([A-Z]|\.)+#", r"#+(([A-Z]|\.)#)+##*"]:
        line_no, line = next(lines)
        match = re.fullmatch(pattern, line)
        assert match, f"parse error line {line_no}; expected {pattern!r}"
        if line_no == 1:
            hallway = line
        elif line_no == 2:
            hallway_bottom = line

    assert len(hallway) == len(
        hallway_bottom
    ), f"parse error line 2; length mismatch with line 1"
    queue_ixs = [i for i, c in enumerate(hallway_bottom) if c != "#"]
    assert not any(
        hallway[i] != "." for i in queue_ixs
    ), f"parse error line 1; no characters allowed in spaces above rooms"

    left_q_len = queue_ixs[0] - 1
    right_q_len = len(hallway) - queue_ixs[-1] - 2
    middle = "".join(
        "#" * (ix2 - ix1 - 1) + r"([A-Z]|\.)"
        for ix1, ix2 in zip(queue_ixs[:-1], queue_ixs[1:])
    )
    row_pattern = fr"{'(?: |#)' * left_q_len}#([A-Z]|\.){middle}#+"
    end_pattern = (
        fr"{'(?: |#)' * left_q_len}{'#' * (len(hallway) - left_q_len - right_q_len)}#*"
    )

    room_queues = [
        [None if hallway_bottom[i] == "." else hallway_bottom[i]] for i in queue_ixs
    ]
    frozen = [False] * len(room_queues)
    for line_no, line in lines:
        if re.fullmatch(end_pattern, line):
            break
        match = re.fullmatch(row_pattern, line)
        assert match, f"parse error line {line_no}; expected {row_pattern!r}"
        elements = match.groups()
        for i, (elt, q, fr) in enumerate(zip(elements, room_queues, frozen)):
            if elt == "#":
                frozen[i] = True
            else:
                assert not fr, f"broken column, {elt!r} in column {i}, line {line_no}"
                q.append(None if elt == "." else elt)

    room_queues = list(map(list, map(reversed, room_queues)))

    chars = [
        c for c in islice(map(chr, range(ord("A"), ord("Z") + 1)), len(room_queues))
    ]
    end_char_ixs = {c: 2 * i + 1 for i, c in enumerate(chars)}
    costs = dict(zip(chars, map((10).__pow__, range(len(chars)))))
    counts = Counter(filter(not_null, chain.from_iterable(room_queues)))
    bad_chars = set(counts).difference(chars)
    assert (
        not bad_chars
    ), f"parsed characters that can't be mapped to rooms: {bad_chars}; should be 'A', 'B', ..."

    for c, i in zip(chars, range(len(room_queues))):
        assert counts[c] <= len(
            room_queues[i]
        ), f"too many instances of {c!r}; got {counts[c]}, expected at most {len(room_queues[i])} (size of room {i})"

    n_queues = 2 * len(room_queues) + 1
    queues = []
    edges = []
    for i in range(n_queues):
        # rooms are odd
        if i == 0:
            q = [None if c == "." else c for c in hallway[1 : left_q_len + 1]]
        elif i == n_queues - 1:
            q = [None if c == "." else c for c in hallway[-2 : -right_q_len - 2 : -1]]
        elif i % 2 == 1:
            q = room_queues[(i - 1) // 2]
        # hallway ends and stopping-places between room are odd
        else:
            c = hallway[queue_ixs[(i - 1) // 2] + 1]
            q = [None if c == "." else c]

        queues.append(q)
        edges_to_left = zip(
            range(i - 1, -1, -1),
            islice(
                chain.from_iterable(repeat(2 * i + 1, 2) for i in range(n_queues)),
                i % 2,
                None,
            ),
        )
        edges_to_right = zip(
            range(i + 1, n_queues),
            islice(
                chain.from_iterable(repeat(2 * i + 1, 2) for i in range(n_queues)),
                i % 2,
                None,
            ),
        )
        edges.append([*reversed(list(edges_to_left)), *edges_to_right])

    start_state = QueueNetwork(queues, edges, costs.__getitem__)
    return start_state, end_char_ixs


# Output


def tostr(state: QueueNetwork) -> str:
    width = len(state.queues) + len(state.queues[0]) + len(state.queues[-1])
    lines = ["#" * width]
    middle = ".".join("." if q[0] is None else q[0] for q in state.queues[2:-1:2])
    hall = (
        f"#{''.join('.' if c is None else c for c in state.queues[0])}"
        f".{middle}."
        f"{''.join('.' if c is None else c for c in reversed(state.queues[-1]))}#"
    )
    lines.append(hall)
    otherlines = []
    maxsize = max(map(len, state.queues))
    for i in range(maxsize):
        if i == maxsize - 1:
            l, r = "#" * (len(state.queues[0]) + 1), "#" * (len(state.queues[-1]) + 1)
        else:
            l, r = " " * len(state.queues[0]) + "#", "#" + " " * len(state.queues[-1])

        middle = "#".join(
            (q[i] or ".") if len(q) > i else "." for q in state.queues[1:-1:2]
        )
        otherlines.append(f"{l}{middle}{r}")

    lines.extend(reversed(otherlines))
    lines.append(
        f"{' ' * len(state.queues[0])}{'#' * (width - len(state.queues[0]) - len(state.queues[-1]))}{' ' * len(state.queues[-1])}"
    )
    return "\n".join(lines)


def animate(
    path: List[QueueNetwork[str, int]],
    filename: str,
    block_size: int = 25,
    frame_duration_ms: int = 500,
    bg_color: str = "#000000",
    wall_color: str = "#2f4f4f",
    colors=[
        "#2ca02c",
        "#d62728",
        "#1f77b4",
        "#ffbb78",
        "#ff7f0e",
        "#9467bd",
        "#e377c2",
        "#aec7e8",
        "#bcbd22",
        "#17becf",
    ],
):
    import numpy as np
    from PIL import Image

    def fromhex(color: str) -> Tuple[int, int, int]:
        color = color.lstrip("#")
        return (int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16))

    def to_image(state: QueueNetwork[str, int]) -> Image.Image:
        lines = tostr(state).rstrip().splitlines()
        w = max(map(len, lines))
        h = len(lines)
        im = np.empty((h, w, 3), dtype="uint8")
        for i, c in enumerate(fromhex(bg_color)):
            im[:, :, i] = c

        for i, line in enumerate(lines):
            im[i, : len(line), :] = list(map(color_lookup.__getitem__, line))

        return Image.fromarray(im, mode="RGB").resize(
            (w * block_size, h * block_size), Image.BOX
        )

    def to_gif(ims: List[Image.Image]):
        ims[0].save(
            filename,
            format="GIF",
            append_images=ims[1:],
            save_all=True,
            duration=frame_duration_ms,
            loop=0,
        )

    color_lookup = {
        chr(i): fromhex(color)
        for i, color in zip(range(ord("A"), ord("Z") + 1), colors)
    }
    color_lookup[" "] = fromhex(bg_color)
    color_lookup["."] = fromhex(bg_color)
    color_lookup["#"] = fromhex(wall_color)

    ims = list(map(to_image, path))
    print(
        f"Writing {frame_duration_ms * len(ims) / 1000:3.2f}s GIF with {len(ims):d} frames to {filename}"
    )
    to_gif(ims)


# Problem-specific helpers


def is_end_state(end_char_ixs: Dict[str, int], state: QueueNetwork[str, int]) -> bool:
    rooms = state.queues[1::2]
    nonrooms = state.queues[0::2]
    return all(all(map(is_null, q)) for q in nonrooms) and all(
        all(is_null(c) or end_char_ixs[c] == i for c in q)
        for i, q in zip(range(1, len(state.queues), 2), rooms)
    )


def gen_states(
    end_char_ixs: Dict[str, int], state: QueueNetwork[str, int]
) -> Iterator[Tuple[QueueNetwork[str, int], int]]:
    for new_state, char, from_, to, cost in state.moves():
        # blocked
        start, end = min(from_, to), max(from_, to)
        first_blocking_index = start + 2 - start % 2
        final_moves = []
        intermediate_moves = []
        if any(any(map(not_null, q)) for q in state.queues[first_blocking_index:end:2]):
            print_(f"skipping {char!r} {from_} -> {to}; blocked")
            pass
        # "Amphipods will never move from the hallway into a room unless that room is their
        # destination room and that room contains no amphipods which do not also have that room
        # as their own destination."
        elif to % 2 == 1:
            if to == end_char_ixs[char] and all(
                c == char or is_null(c) for c in state.queues[to]
            ):
                print_(
                    f"move from {'hall' if from_ % 2 == 0 else 'room'} to room, {char!r} {from_} -> {to}"
                )
                final_moves.append((new_state, cost))
            else:
                print_(
                    f"skipping {char!r} {from_} -> {to}; can't move into wrong or occupied room"
                )
                pass
        # "Once an amphipod stops moving in the hallway, it will stay in that spot until it can
        # move into a room."
        elif from_ % 2 == 0:
            # not moving into a room, else we would have entered last `if`
            print_(
                f"skipping {char!r} {from_} -> {to}; must move into room from hallway"
            )
            pass
        # any other transition
        else:
            intermediate_moves.append((new_state, (char, from_, to, cost)))

        if final_moves:
            yield from final_moves
        else:
            for move in intermediate_moves:
                (new_state, (char, from_, to, cost)) = move
                print_(f"move from room to hall, {char!r} {from_} -> {to}")
                yield new_state, cost


# Tests


test_case1 = """#############
#...........#
###B#C#B#D###
  #A#D#C#A#
  #########"""

test_cost1 = 12521

test_case2 = """#############
#.....A....A.#
###.#.#.#.####
  #.#.#A#.#
  #.#.#.#.#
  #########"""

test_cost2 = 23


def test():
    print("running tests")
    for test_case, test_cost in [(test_case1, test_cost1), (test_case2, test_cost2)]:
        lines = test_case.splitlines()
        start_state, end_char_ixs = parse_input(lines)
        transition_graph = LazyDataGraph(partial(gen_states, end_char_ixs))
        end_state = partial(is_end_state, end_char_ixs)
        path, cost = djikstra(
            transition_graph, start_state, end_state, cost_fn=identity
        )
        assert (
            cost == test_cost
        ), f"failed test:\n\n{test_case}\n\nexpected cost {test_cost}, computed cost {cost}"


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and "-v" in sys.argv or "--verbose" in sys.argv:
        VERBOSE = True

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test()

    if "--gif" in sys.argv:
        import json

        i = sys.argv.index("--gif")
        filename = sys.argv[i + 1] if len(sys.argv) > i + 1 else "day23.gif"
        options = json.loads(sys.argv[i + 2]) if len(sys.argv) > i + 2 else {}
        gif = True
    else:
        gif = False
        filename = None
        options = None

    with open("day23.txt") if sys.stdin.isatty() else sys.stdin as f:
        lines = list(f)
        start_state1, end_char_ixs = parse_input(lines)
        transition_graph = LazyDataGraph(partial(gen_states, end_char_ixs))
        end_state = partial(is_end_state, end_char_ixs)
        if sys.stdin.isatty():
            part2 = True
            start_state2, _ = parse_input(
                lines[:3] + ["  #D#C#B#A#", "  #D#B#A#C#"] + lines[-2:]
            )
        else:
            part2 = False

    # Part 1

    print(start_state1, end="\n\n")
    path, cost = djikstra(transition_graph, start_state1, end_state, cost_fn=identity,)
    for p in path[1:]:
        print(p, end="\n\n")
    print(cost)

    if gif and not part2:
        animate(path, filename, **options)

    # Part 2

    if part2:
        print("\n\n\n")
        print(start_state2, end="\n\n")
        path, cost = djikstra(
            transition_graph, start_state2, end_state, cost_fn=identity,
        )
        for p in path[1:]:
            print(p, end="\n\n")
        print(cost)

        if gif:
            animate(path, filename, **options)
