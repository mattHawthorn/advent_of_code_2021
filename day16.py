from functools import partial, reduce, singledispatch
from itertools import chain
from operator import mul, gt, lt, eq
from typing import (
    ByteString,
    Callable,
    Iterable,
    List,
    NamedTuple,
    Tuple,
    TypeVar,
    Union,
)

Packet = ByteString
Byte = int
Version = int
MessageType = int
T = TypeVar("T")
Parser = Callable[["BitBufferView"], Tuple["Message", "BitBufferView"]]


class Message:
    pass


class Reduce(NamedTuple, Message):
    packets: List[Packet]
    func: Callable[[Iterable[int]], int]

    @property
    def value(self) -> int:
        return self.func(p.value for p in self.packets)


class BinOp(NamedTuple, Message):
    packets: List[Packet]
    op: Callable[[int, int], int]

    @property
    def value(self) -> int:
        return self.op(self.packets[0].value, self.packets[1].value)


class Literal(NamedTuple, Message):
    value: int


class Packet(NamedTuple):
    version: Version
    message: Message

    @property
    def value(self) -> int:
        return self.message.value


def parse(packet_hex: str):
    return _parse_packet(BitBufferView(bytes.fromhex(packet_hex)))


@singledispatch
def version_sum(_) -> int:
    return 0


@version_sum.register(Packet)
def version_sum_packet(packet: Packet) -> int:
    return packet.version + version_sum(packet.message)


@version_sum.register(Reduce)
@version_sum.register(BinOp)
def version_sum_op(op: Union[Reduce, BinOp]) -> int:
    return sum(map(version_sum_packet, op.packets))


def identity(x: T) -> T:
    return x


class BitBufferView:
    def __init__(self, data: ByteString, offset=0):
        self.data = data
        self.offset = offset

    def read(
        self, n_bits: int, f: Callable[[bytes], T] = identity
    ) -> Tuple[int, "BitBuffer"]:
        new_buffer = BitBufferView(self.data, self.offset + n_bits)
        bit_offset_start, bit_offset_end = self.offset, (self.offset + n_bits)
        byte_offset_start, trim_bits_left = divmod(bit_offset_start, 8)
        byte_offset_end, _ = divmod(bit_offset_end, 8)
        trim_bits_right = 8 - _
        if byte_offset_start == byte_offset_end:
            byte_ = self.data[byte_offset_start]
            trimmed = (byte_ & _byte_masks[trim_bits_left]) >> trim_bits_right
            read_bytes = bytes([trimmed])
        else:
            bytes_ = self.data[byte_offset_start : byte_offset_end + 1]
            first_bytes = chain(
                (next(iter(bytes_)) & _byte_masks[trim_bits_left],), bytes_[1:]
            )
            read_bytes = bytes(
                map(partial(one_byte, trim_bits_right), first_bytes, bytes_[1:])
            )

        return f(read_bytes), new_buffer

    def can_read(self, n_bits: int):
        return (self.offset + n_bits) <= (8 * len(self.data))


# mask off the left n bits using `&`, where n is the index in this list
_byte_masks = [(2 ** i) - 1 for i in range(8, -1, -1)]


def one_byte(trim_bits_right: int, b1: Byte, b2: Byte) -> Byte:
    trim_left = 8 - trim_bits_right
    return ((b1 & _byte_masks[trim_left]) << trim_left) ^ (b2 >> trim_bits_right)


to_int = partial(int.from_bytes, byteorder="big")
prod = partial(reduce, mul)


def _parse_packet(buffer: BitBufferView) -> Tuple[Packet, BitBufferView]:
    version, buffer = buffer.read(3, ord)
    type_, buffer = buffer.read(3, ord)
    parser = _parser_for(type_)
    value, buffer = parser(buffer)
    return Packet(version, value), buffer


def _parse_literal(buffer: BitBufferView) -> Tuple[Literal, BitBufferView]:
    value = 0
    while buffer.can_read(5):
        hexdigit, buffer = buffer.read(5, ord)
        i = hexdigit & 15
        value = value * 16 + i
        if (hexdigit >> 4) == 0:
            # last byte
            break

    return Literal(value), buffer


def _parse_op(
    type_: MessageType, buffer: BitBufferView
) -> Tuple[Union[BinOp, Reduce], BitBufferView]:
    len_type, buffer = buffer.read(1, ord)
    if len_type == 0:
        num_bits, buffer = buffer.read(15, to_int)
        packets = []
        bits_read = 0
        while bits_read < num_bits:
            packet, buffer_ = _parse_packet(buffer)
            bits_read += buffer_.offset - buffer.offset
            buffer = buffer_
            packets.append(packet)
    else:
        num_packets, buffer = buffer.read(11, to_int)
        packets = []
        for _ in range(num_packets):
            packet, buffer = _parse_packet(buffer)
            packets.append(packet)

    op = _opfuncs[type_]
    if type_ <= 3:
        return Reduce(packets, op), buffer
    else:
        return BinOp(packets, op), buffer


_opfuncs = [sum, prod, min, max, None, gt, lt, eq]


def _parser_for(type_: MessageType) -> Parser:
    if type_ == 4:
        return _parse_literal
    return partial(_parse_op, type_)


def test():
    print("running tests")
    packet, buffer = parse("D2FE28")
    assert packet.version == 6
    assert version_sum(packet) == 6
    assert packet.value == 2021

    packet, buffer = parse("8A004A801A8002F478")
    assert version_sum(packet) == 16

    packet, buffer = parse("9C0141080250320F1802104A08")
    assert packet.value == 1


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test()

    with open("day16.txt") if sys.stdin.isatty() else sys.stdin as f:
        hex_ = f.read().strip()

    # Part 1

    packet, buffer = parse(hex_)
    print(version_sum(packet))

    # Part 2

    print(packet.value)
