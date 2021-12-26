from collections import defaultdict
from functools import partial
from typing import AbstractSet, Callable, Hashable, Iterator, Mapping, List, TypeVar

T = TypeVar("T", bound=Hashable)
Graph = Mapping[T, AbstractSet[T]]
Path = List[T]


# Main function


def all_paths(
    graph: Graph[T], path: Path[T], end: T, can_visit: Callable[[Path[T], T], bool],
) -> Iterator[Path[T]]:
    tail = path[-1]
    if tail == end:
        yield path
    else:
        next_nodes = filter(partial(can_visit, path), graph[tail])
        for node in next_nodes:
            yield from all_paths(
                graph, path + [node], end, can_visit,
            )


# Problem-specific helpers

from collections import Counter

START, END = "start", "end"


def can_visit_part1(path: Path[str], node: str) -> bool:
    small = node.islower()
    return node != START and (not small or (small and node not in path))


def can_visit_part2(path: Path[str], node: str) -> bool:
    if node == START:
        return False
    elif node.islower():
        if node not in path:
            return True
        else:
            small_counts = Counter(filter(str.islower, path))
            return small_counts[node] < 2 and max(small_counts.values()) < 2
    else:
        return True


# Input parsing


def parse_graph(f):
    graph = defaultdict(set)
    for line in f:
        head, tail = line.strip().split("-")
        graph[head].add(tail)
        graph[tail].add(head)
    return graph


def test():
    print("running tests")
    from io import StringIO

    f = StringIO(
        """
    dc-end
    HN-start
    start-kj
    dc-start
    dc-HN
    LN-dc
    HN-end
    kj-sa
    kj-HN
    kj-dc
    """.strip()
    )
    graph = parse_graph(f)

    assert sum(1 for p in all_paths(graph, [START], END, can_visit_part1)) == 19
    assert sum(1 for p in all_paths(graph, [START], END, can_visit_part2)) == 103


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test()

    with open("day12.txt") if sys.stdin.isatty() else sys.stdin as f:
        graph = parse_graph(f)

    # Part 1

    paths = all_paths(graph, [START], END, can_visit_part1)
    print(sum(1 for p in paths))

    # Part 2

    paths = all_paths(graph, [START], END, can_visit_part2)
    print(sum(1 for p in paths))
