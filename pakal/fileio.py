import io
import os
import tempfile
from collections.abc import Iterator
from contextlib import AbstractContextManager
from types import TracebackType
from typing import IO, AnyStr, Optional, Union, overload

import numpy as np
from numpy.typing import ArrayLike


class ResourceStream(io.RawIOBase):
    __slots__ = ('_pos', '_size', '_res')

    def __init__(self, res: 'ResourceFile') -> None:
        self._pos = 0
        self._size = len(res.buffer)
        self._res = res

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == io.SEEK_CUR:
            offset += self._pos
        elif whence == io.SEEK_END:
            offset += self._size
        self._pos = offset
        return self._pos

    def tell(self) -> int:
        return self._pos

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def read(self, size: int | None = None) -> bytes:
        if size is not None and size >= 0:
            size = min(self._size - self._pos, size)
        else:
            size = self._size - self._pos
        end = (self._pos + size) if size is not None else None
        return self._res.buffer[self._pos:end].tobytes()

    def readinto(self, b: bytearray) -> int: # type: ignore[override]
        size = min(len(b), self._size - self._pos)
        b[:size] = self._res.buffer[self._pos:self._pos + size]
        self._pos += size
        return size

    def close(self) -> None:
        super().close()
        self._res.close()

    def partial(self, offset: int, size: int) -> IO[bytes]:
        if size > io.DEFAULT_BUFFER_SIZE:
            res = ResourceFile(self._res[offset:offset + size])
            return io.BufferedReader(ResourceStream(res))
        return io.BytesIO(self._res[offset:offset + size]) # type: ignore[arg-type]


class ResourceFile(AbstractContextManager[memoryview]):
    __slots__ = ('buffer', 'closed')

    def __init__(self, buffer: ArrayLike, tmpfile: Optional[tempfile.TemporaryFile] = None) -> None:
        self.buffer = memoryview(buffer)  # type: ignore[arg-type]
        self.closed = False
        self._tmpfile = tmpfile

    def __len__(self) -> int:
        return len(self.buffer)

    def __buffer__(self, _flags: int) -> memoryview:
        return self.buffer

    def __exit__(
        self,
        __exc_type: type[BaseException] | None,
        __exc_value: BaseException | None,
        __traceback: TracebackType | None,
    ) -> bool | None:
        self.close()
        return None

    @overload
    def __getitem__(self, index: slice) -> ArrayLike: ...
    @overload
    def __getitem__(self, index: int) -> int: ...
    def __getitem__(self, index: slice | int) -> ArrayLike | int:
        if not self.closed:
            return self.buffer[index]
        raise OSError('I/O operation on closed file')  # noqa: TRY003

    @classmethod
    def load(
        cls,
        file_path: Union[str, bytes, os.PathLike[AnyStr]],
        key: int = 0x00,
    ) -> 'ResourceFile':
        data = np.memmap(file_path, dtype='u1', mode='r')

        if key == 0x00:
            return cls(data)

        tmp = tempfile.TemporaryFile()
        result = np.memmap(tmp, dtype='u1', mode='w+', shape=data.shape)
        result[:] = data ^ key
        result.flush()
        del result
        return cls(np.memmap(tmp, dtype='u1', mode='r', shape=data.shape), tmp)

    def close(self) -> None:
        if self._tmpfile is not None:
            self._tmpfile.close()
            self._tmpfile = None
        self.closed = True

    @classmethod
    def open(
        cls,
        file: Union[str, bytes, os.PathLike[AnyStr]],
        mode: str = 'r',
        encoding: str = 'utf-8',
        errors: Optional[str] = None,
    ) -> Iterator[IO[AnyStr]]:
        ostream: IO  # type: ignore[type-arg]
        res = cls.load(file)
        ostream = io.BufferedReader(ResourceStream(res))
        if 'b' not in mode:
            ostream = io.TextIOWrapper(
                ostream,
                encoding=encoding,
                errors=errors,
            )
        return ostream


def read_file(file_path: str, key: int = 0x00) -> bytes:
    with ResourceFile.load(file_path, key=key) as res:
        return bytes(res)
