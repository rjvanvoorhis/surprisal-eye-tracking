import typing
import uuid

T = typing.TypeVar("T")
P = typing.ParamSpec("P")
MODEL_NAME = "bert-base-uncased"

_CACHE = {}


def cache_result(func: typing.Callable[P, T]) -> typing.Callable[P, T]:
    key = uuid.uuid4()

    def decorated(*args: P.args, **kwargs: P.kwargs) -> T:
        value: typing.Optional[T] = _CACHE.get(key)
        if value is None:
            value = func(*args, **kwargs)
            _CACHE[key] = value
        return value

    return decorated
