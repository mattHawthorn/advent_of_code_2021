from collections import defaultdict
from functools import partial
from itertools import starmap
from operator import not_
from typing import Iterable, List, Mapping, NamedTuple, Optional, Tuple


class BoardState:
    def __init__(self, rows: List[List[int]]):
        self.rows = rows
        self.num_to_ix = num_to_ixs(rows)
        self.row_ixs = {i: set() for i in range(5)}
        self.col_ixs = {i: set() for i in range(5)}
        self.win_rows = set()
        self.win_cols = set()

    @property
    def wins(self):
        return bool(self.win_rows) or bool(self.win_cols)

    def mark(self, num: int):
        coords = self.num_to_ix.get(num, [])
        for x, y in coords:
            self._mark(x, y)

    def marked(self, x, y) -> bool:
        row_ixs = self.row_ixs[x]
        col_ixs = self.col_ixs[y]
        return y in row_ixs or x in col_ixs

    def _mark(self, x: int, y: int):
        row_ixs = self.row_ixs[x]
        col_ixs = self.col_ixs[y]
        row_ixs.add(y)
        col_ixs.add(x)
        if len(row_ixs) > 4:
            self.win_rows.add(x)
        if len(col_ixs) > 4:
            self.win_cols.add(y)

    @property
    def score(self):
        return sum(
            n * sum(map(not_, starmap(self.marked, coords)))
            for n, coords in self.num_to_ix.items()
        )

    def __str__(self):
        def formatnum(n, marked):
            return f"[{n: 3d}]" if marked else f" {n: 3d} "

        return "\n".join(
            " ".join(formatnum(n, self.marked(i, j)) for j, n in enumerate(row))
            for i, row in enumerate(self.rows)
        )

    __repr__ = __str__


def num_to_ixs(board: List[List[int]]) -> Mapping[int, List[Tuple[int, int]]]:
    num_to_ix = defaultdict(list)
    for i, row in enumerate(board):
        for j, num in enumerate(row):
            num_to_ix[num].append((i, j))

    return num_to_ix


class Win(NamedTuple):
    board_idx: int
    board: BoardState
    last_num: int
    last_turn: int


class GameState:
    def __init__(self, boards):
        self.boards = list(map(BoardState, boards))

    def play(self, nums: Iterable[int]) -> Optional[Win]:
        for turn, num in enumerate(nums):
            for i, board in enumerate(self.boards):
                board.mark(num)
                if board.wins:
                    return Win(i, board, num, turn)
        return Win(None, None, num, turn)

    def play_all(self, nums: Iterable[int]) -> List[Win]:
        wins = []
        winning_ixs = set()
        for turn, num in enumerate(nums):
            if len(winning_ixs) == len(self.boards):
                break
            for i, board in enumerate(self.boards):
                board.mark(num)
                if i not in winning_ixs:
                    if board.wins:
                        winning_ixs.add(i)
                        wins.append(Win(i, board, num, turn))
        return wins


# Input parsing


def parse_bingo(f):
    line1 = next(f).strip()
    nums = list(map(int, line1.split(",")))

    def parse_line(line: str):
        return list(map(int, line.strip().split()))

    def parse_board(f, lines=None):
        if not lines:
            newlines = filter(bool, map(str.strip, iter(f.readline, "")))
            line = next(newlines, None)
            lines = []
        elif len(lines) == 5:
            return lines
        else:
            line = next(f, None)
        if not line:
            return None
        lines.append(parse_line(line))
        return parse_board(f, lines)

    boards = list(iter(partial(parse_board, f), None))
    return nums, boards


if __name__ == "__main__":
    import sys

    with open("day4.txt") if sys.stdin.isatty() else sys.stdin as f:
        nums, boards = parse_bingo(f)

    # Part 1

    game = GameState(boards)
    result = game.play(nums)
    print(result.board.score * result.last_num)

    # Part 2

    game = GameState(boards)
    results = game.play_all(nums)
    result = results[-1]
    print(result.board.score * result.last_num)
