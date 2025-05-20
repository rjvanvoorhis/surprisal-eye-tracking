import typing

T = typing.TypeVar("T")


class SizedIterable(typing.Protocol[T]):
    def __iter__(self) -> typing.Iterable[T]: ...
    def __len__(self) -> int: ...
