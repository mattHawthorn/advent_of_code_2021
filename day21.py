from collections import Counter, defaultdict, deque
from itertools import accumulate, cycle, islice, product, repeat, tee
from typing import Callable, Iterable, List, Mapping, Tuple, TypeVar

BoardPosition = int
Score = int
DiceRoll = int
PlayerState = Tuple[BoardPosition, Score]
StateCounts = Mapping[PlayerState, int]
T = TypeVar("T")


def turn(player: PlayerState, roll: DiceRoll, board_size: int = 10) -> PlayerState:
    position, score = player
    new_position = ((position - 1 + roll) % board_size) + 1
    return new_position, score + new_position


def play(
    players: List[PlayerState], rolls: Iterable[DiceRoll], max_score: int
) -> Tuple[int, List[PlayerState]]:
    n_turns = 0
    rolls = iter(rolls)
    states = deque(players, maxlen=len(players))
    while states[-1][1] < max_score:
        states.append(turn(states.popleft(), next(rolls)))
        n_turns += 1

    offset = n_turns % len(players)
    states = list(states)
    # put players back in original order
    return n_turns, [*states[-offset:], *states[:offset]]


def chunked(size: int, iterable: Iterable[T]) -> Iterable[List[T]]:
    it = iter(iterable)
    yield from iter(lambda: list(islice(it, size)), [])


def deterministic_dice(n_sides: int) -> Iterable[DiceRoll]:
    return cycle(range(1, n_sides + 1))


def dirac(
    board_size: int,
    max_score: int,
    n_sides: int,
    n_rolls_per_turn: int,
    states: List[PlayerState],
) -> List[Tuple[int, int]]:
    all_sides = range(1, n_sides + 1)
    roll_frequencies = Counter(map(sum, product(*([all_sides] * n_rolls_per_turn))))
    n_players = len(states)
    player_lose_counts, player_win_counts = zip(
        *[
            winning_and_losing_counts(board_size, max_score, roll_frequencies, state)
            for state in states
        ]
    )
    # number of ways for player i to win in n turns while all other players fail to win
    # in n or n - 1 turns
    win_counts = defaultdict(int)
    lose_counts = defaultdict(int)

    def losses_at_turn(turn: int, player: int):
        lose_counts = player_lose_counts[player]
        return lose_counts[turn] if 0 <= turn < len(lose_counts) else 0

    def wins_at_turn(turn: int, player: int):
        win_counts = player_win_counts[player]
        return win_counts[turn] if 0 <= turn < len(win_counts) else 0

    max_turns = max(map(len, player_win_counts))
    for turn_no in range(max_turns):
        for player_no in range(n_players):
            # wins
            n_wins = wins_at_turn(turn_no, player_no)
            n_other_losses = 1
            for other_player_no in range(n_players):
                if other_player_no < player_no:
                    n_other_losses *= losses_at_turn(turn_no, other_player_no)
                elif other_player_no > player_no:
                    n_other_losses *= losses_at_turn(turn_no - 1, other_player_no)

            total_games = n_wins * n_other_losses
            win_counts[player_no] += total_games
            # losses
            for other_player_no in filter(player_no.__ne__, range(n_players)):
                lose_counts[other_player_no] += total_games

    return [(win_counts.get(i, 0), lose_counts.get(i, 0)) for i in range(n_players)]


def winning_and_losing_counts(
    board_size: int,
    max_score: int,
    roll_frequencies: Mapping[DiceRoll, int],
    start_state: PlayerState,
) -> Tuple[List[int], List[int]]:
    states_at_n: List[int] = [1]
    winning_at_n: List[int] = [0]
    new_states = {start_state: 1}
    while new_states:
        new_states, winning = scores_at_n(
            board_size, max_score, roll_frequencies, new_states
        )
        winning_at_n.append(sum(winning.values()))
        states_at_n.append(sum(new_states.values()))

    return states_at_n, winning_at_n


def scores_at_n(
    board_size: int,
    max_score: int,
    roll_frequencies: Mapping[int, int],
    states_at_n_minus_1: StateCounts,
) -> Tuple[StateCounts, StateCounts]:
    new = defaultdict(int)
    winning = defaultdict(int)
    for state, count in states_at_n_minus_1.items():
        for roll, frequency in roll_frequencies.items():
            new_state = turn(state, roll, board_size)
            update = winning if new_state[1] >= max_score else new
            update[new_state] += count * frequency

    return new, winning


if __name__ == "__main__":
    import sys

    with open("day21.txt") if sys.stdin.isatty() else sys.stdin as f:
        positions = [
            int(line.split(" ")[-1]) for line in filter(bool, map(str.strip, f))
        ]
        players = [(p, 0) for p in positions]

    # Part 1

    n_rolls = 3
    rolls = map(sum, chunked(n_rolls, deterministic_dice(100)))
    final_turn, final_state = play(players, rolls, max_score=1000)
    losing_score = min(score for _, score in final_state)
    print(final_turn * n_rolls, "rolls, final state:", final_state)
    print(final_turn * losing_score * n_rolls)

    # Part 2

    win_lose_counts = dirac(
        board_size=10, max_score=21, n_sides=3, n_rolls_per_turn=3, states=players
    )
    print("Final win/loss counts:", win_lose_counts)
    print(max(wins for wins, losses in win_lose_counts))
