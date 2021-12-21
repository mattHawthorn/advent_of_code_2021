from functools import reduce
from typing import Iterable, Callable, Tuple, TypeVar

Direction = str
Coord = Tuple[int, int]
Aim = int
Instruction = Tuple[Direction, int]
State = TypeVar("State")

F, D, U = "forward", "down", "up"


def navigate(instructions: Iterable[Instruction], move: Callable[[State, Instruction], State], start: State) -> State:
    return reduce(move, instructions, start)


def move(location: Coord, instruction: Instruction) -> Coord:
    direction, magnitude = instruction
    if direction == F:
        return (location[0] + magnitude, location[1])
    else:
        sign = [U, None, D].index(direction) - 1
        return (location[0], location[1] + magnitude * sign)


def move_aimed(state: State, instruction: Instruction) -> State:
    location, aim = state
    direction, magnitude = instruction
    if direction == F:
        return (location[0] + magnitude, location[1] + aim * magnitude), aim
    else:
        sign = [U, None, D].index(direction) - 1
        return location, aim + magnitude * sign


# Input parsing

def parse_instruction(s: str) -> Instruction:
    direction, magnitude = s.strip().split()
    return direction, int(magnitude)


if __name__ == "__main__":
    import sys

    with open("day2.txt") if sys.stdin.isatty() else sys.stdin as f:
        instructions = list(map(parse_instruction, filter(bool, map(str.strip, f))))

    # Part 1

    (x, y) = navigate(instructions, move, (0, 0))
    print(x * y)

    # Part 2

    (x, y), _ = navigate(instructions, move_aimed, ((0, 0), 0))
    print(x * y)
