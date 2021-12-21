from copy import deepcopy
from functools import partial, reduce
from itertools import accumulate, repeat, takewhile
from heapq import heapify, heappop, heappush
from typing import Callable, Iterator, List, Optional, Set, Tuple, TypeVar, Union

EXPLODE_DEPTH_THRESHOLD = 4
SPLIT_VALUE_THRESHOLD = 9

T = TypeVar("T")
Leaf = int
Tree = List[Union[Leaf, "Tree"]]
L, R = False, True


class Address(Tuple[bool, ...]):
    def __new__(cls, *args):
        return super().__new__(cls, args)

    def __getitem__(self, item):
        new = super().__getitem__(item)
        return Address(*new) if isinstance(item, slice) else new

    def __lt__(self, other):
        # compare addresses in left-to-right dfs order
        otherlen = len(other)
        if otherlen == 0:
            # root
            return bool(self) and self[0] == L
        elif otherlen <= len(self) and self[:otherlen] == other:
            # ancestor
            return self[otherlen:] < ()
        else:
            # base case
            return super().__lt__(other)


def iterate(f: Callable[[T], T], x: T) -> Iterator[T]:
    def call(x, f):
        return f(x)

    return accumulate(repeat(f), call, initial=x)


def dfs(
    tree, address: Optional[Address] = None
) -> Iterator[Tuple[Address, Union[Tree, Leaf]]]:
    address = address or Address()
    yield address, tree
    if not isinstance(tree, Leaf):
        yield from dfs(tree[0], Address(*address, L))
        yield from dfs(tree[1], Address(*address, R))


def get_subtree(
    address: Address, tree: Union[Tree, Leaf]
) -> Optional[Union[Tree, Leaf]]:
    if not address:
        return tree
    elif isinstance(tree, Leaf):
        # address is not present
        return None
    else:
        root = tree[address[0]]
        return get_subtree(address[1:], root)


def set_subtree(address: Address, tree: Tree, subtree: Union[Tree, Leaf]) -> Tree:
    if not address:
        return subtree
    root = get_subtree(address[:-1], tree)
    root[address[-1]] = subtree
    return tree


def cousin_leaf_on_side(
    side: bool, tree: Tree, address: Address
) -> Optional[Tuple[Address, Leaf]]:
    ascend_n = sum(1 for _ in takewhile(side.__eq__, reversed(address)))
    if ascend_n == len(address):
        return None
    new_address = Address(*address[: -ascend_n - 1], side)
    root = get_subtree(new_address, tree)
    if root is None:
        return None
    down1 = Address(not side)
    subtrees = iterate(partial(get_subtree, down1), root)
    for steps_down, subtree in enumerate(subtrees):
        if isinstance(subtree, Leaf):
            return Address(*new_address, *(down1 * steps_down)), subtree


def reduce_step(
    explodeq: List[Address], splitq: List[Address], splitset: Set[Address], tree: Tree
) -> Tree:
    if explodeq:
        explode_addr = heappop(explodeq)
        explode_tree = get_subtree(explode_addr, tree)
        print_("explode", explode_tree, "at", explode_addr)
        for side in L, R:
            leaf_address_leaf = cousin_leaf_on_side(
                side, tree, Address(*explode_addr, side)
            )
            if leaf_address_leaf is not None:
                leaf_address, leaf = leaf_address_leaf
                new_leaf = leaf + explode_tree[side]
                print_("put", new_leaf, leaf_address)
                set_subtree(leaf_address, tree, new_leaf)
                if new_leaf > SPLIT_VALUE_THRESHOLD and leaf_address not in splitset:
                    splitset.add(leaf_address)
                    heappush(splitq, leaf_address)
        set_subtree(explode_addr, tree, 0)
    elif splitq:
        while splitq:
            split_addr = heappop(splitq)
            splitset.remove(split_addr)
            split_leaf = get_subtree(split_addr, tree)
            if split_leaf is not None:
                break
        else:
            print_("END:", tree)
            return tree

        print_("split", split_leaf, "at", split_addr)
        half, rem = divmod(split_leaf, 2)
        l, r = half, half + rem
        set_subtree(split_addr, tree, [l, r])
        if len(split_addr) >= EXPLODE_DEPTH_THRESHOLD:
            heappush(explodeq, split_addr)
        for side, value in [(L, l), (R, r)]:
            if value > SPLIT_VALUE_THRESHOLD:
                new_split_addr = Address(*split_addr, side)
                if new_split_addr not in splitset:
                    splitset.add(new_split_addr)
                    heappush(splitq, new_split_addr)

    print_("END:", tree)
    return tree


def get_reducer(
    tree: Tree,
) -> Tuple[List[Address], List[Address], Callable[[Tree], Tree]]:
    splitq = [
        addr
        for addr, tree in dfs(tree)
        if isinstance(tree, Leaf) and tree > SPLIT_VALUE_THRESHOLD
    ]
    splitset = set(splitq)
    heapify(splitq)
    explodeq = [
        addr
        for addr, tree in dfs(tree)
        if len(addr) >= EXPLODE_DEPTH_THRESHOLD
        and isinstance(tree, list)
        and all(isinstance(l, Leaf) for l in tree)
    ]
    heapify(explodeq)
    reduce_step_ = partial(reduce_step, explodeq, splitq, splitset)
    return explodeq, splitq, reduce_step_


def reduce_tree(tree: Tree) -> Tree:
    print_("START:", tree)
    explodeq, splitq, reducer = get_reducer(tree)
    for tree in iterate(reducer, tree):
        if not splitq and not explodeq:
            break
    return tree


def add(tree1: Union[Tree, Leaf], tree2: Union[Tree, Leaf]) -> Tree:
    return reduce_tree([deepcopy(tree1), deepcopy(tree2)])


def magnitude(tree: Union[Tree, Leaf]) -> int:
    if isinstance(tree, Leaf):
        return tree
    return 3 * magnitude(tree[0]) + 2 * magnitude(tree[1])


VERBOSE = False


def print_(*args, **kw):
    if VERBOSE:
        print(*args, **kw)


def test(verbose: bool):
    print("running tests")

    trees = list(
        map(
            parse,
            """
        [[[0,[4,5]],[0,0]],[[[4,5],[2,6]],[9,5]]]
        [7,[[[3,7],[4,3]],[[6,3],[8,8]]]]
        [[2,[[0,8],[3,4]]],[[[6,7],1],[7,[1,6]]]]
        [[[[2,4],7],[6,[0,5]]],[[[6,8],[2,8]],[[2,1],[4,5]]]]
        [7,[5,[[3,8],[1,4]]]]
        [[2,[2,2]],[8,[8,1]]]
        [2,9]
        [1,[[[9,3],9],[[9,0],[0,7]]]]
        [[[5,[7,4]],7],1]
        [[[[4,2],2],6],[8,7]]""".strip().split(),
        )
    )
    expected = parse("[[[[8,7],[7,7]],[[8,6],[7,7]]],[[[0,7],[6,6]],[8,7]]]")
    global VERBOSE
    if verbose:
        VERBOSE = True
    result = reduce(add, trees)
    VERBOSE = False
    assert result == expected, f"{result!r} != {expected!r}"
    assert magnitude(result) == 3488


if __name__ == "__main__":
    import sys

    def parse(s: str) -> Union[Tree, Leaf]:
        return eval(s)

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test("--verbose" in sys.argv or "-v" in sys.argv)

    with open("day18.txt") if sys.stdin.isatty() else sys.stdin as f:
        trees = list(map(parse, f))

    # Part 1

    total = reduce(add, trees)
    print(magnitude(total))

    # Part 2

    from itertools import permutations, starmap

    print(max(map(magnitude, starmap(add, permutations(trees, 2)))))
