from functools import partial, reduce
from typing import Iterable, Iterator, Optional

close_to_open = dict(zip(")]}>", "([{<"))
open_to_close = dict(map(reversed, close_to_open.items()))
scores = {")": 3, "]": 57, "}": 1197, ">": 25137}
score = scores.__getitem__
completion_scores = {")": 1, "]": 2, "}": 3, ">": 4}


def corrupted_chars(s: str) -> Iterator[str]:
    q = []
    for c in s:
        if c in close_to_open:
            if close_to_open[c] != q.pop():
                yield c
        else:
            q.append(c)


def completion(s: str) -> Optional[Iterable[str]]:
    q = []
    for c in s:
        if c in close_to_open:
            if close_to_open[c] != q.pop():
                break  # corrupt
        else:
            q.append(c)
    else:
        return map(open_to_close.__getitem__, reversed(q))


def completion_score(comp: Iterable[str]) -> int:
    def combine_score(score: int, char: str):
        return score * 5 + completion_scores[char]

    return reduce(combine_score, comp, 0)


def first(default, it: Iterator):
    return next(it, default)


def median(it: Iterable[int]) -> int:
    values = sorted(it)
    return values[len(values) // 2]


if __name__ == "__main__":
    import sys

    with open("day10.txt") if sys.stdin.isatty() else sys.stdin as f:
        lines = list(map(str.strip, f))

    # Part 1

    corrupted = map(corrupted_chars, lines)
    first_corrupted = filter(None, map(partial(first, None), corrupted))
    print(sum(map(score, first_corrupted)))

    # Part 2

    completions = filter(None, map(completion, lines))
    scores = map(completion_score, completions)
    print(median(scores))
