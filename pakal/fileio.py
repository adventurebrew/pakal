"""
fileio.py - Memory-Efficient Random Access to Large Files with XOR Support

This module provides specialized file I/O functionality for efficient random access to
large, potentially XOR-encrypted files. The primary goal is to provide fast,
memory-efficient access without copying data until it's actually read, preserving file
positions through memory views.
"""

import io
import os
import tempfile
from contextlib import AbstractContextManager
from types import TracebackType
from typing import IO, AnyStr, Protocol, cast, overload

import numpy as np
from numpy.typing import ArrayLike


class _SupportsToBytes(Protocol):
    """
    A protocol that defines a method to convert an object to bytes.
    """

    def tobytes(self) -> bytes:
        """
        Convert the object to bytes.
        """


class ResourceStream(io.RawIOBase):
    """
    A stream class that wraps around a ResourceFile to provide a file-like interface.
    This class supports seeking, reading, and reading into a buffer.
    """

    __slots__ = ('_pos', '_res', '_size')

    def __init__(self, res: 'ResourceFile') -> None:
        """
        Initialize the ResourceStream with a ResourceFile.

        :param res: The ResourceFile to wrap.
        """
        self._pos = 0
        self._size = len(res.buffer)
        self._res = res

    def seek(self, offset: int, whence: int = 0) -> int:
        """
        Seek to a specific position in the stream.

        :param offset: The offset to seek to.
        :param whence: The reference point for the offset (default is 0).
        :return: The new position in the stream.
        """
        if whence == io.SEEK_CUR:
            offset += self._pos
        elif whence == io.SEEK_END:
            offset += self._size
        self._pos = offset
        if self._pos < 0:
            raise OSError('Negative seek position')  # noqa: TRY003
        return self._pos

    def tell(self) -> int:
        """
        Get the current position in the stream.

        :return: The current position.
        """
        return self._pos

    def readable(self) -> bool:
        """
        Check if the stream is readable.

        :return: True if the stream is readable, False otherwise.
        """
        return True

    def seekable(self) -> bool:
        """
        Check if the stream is seekable.

        :return: True if the stream is seekable, False otherwise.
        """
        return True

    def read(self, size: int | None = None) -> bytes:
        """
        Read bytes from the stream.

        :param size: The number of bytes to read (None reads to the end).
        :return: The bytes read from the stream.
        """
        if size is not None and size >= 0:
            size = min(self._size - self._pos, size)
        else:
            size = self._size - self._pos
        prev = self._pos
        self._pos += size
        return cast('_SupportsToBytes', self._res[prev : self._pos]).tobytes()

    def readinto(self, b: bytearray) -> int:  # type: ignore[override]
        """
        Read bytes into a pre-allocated buffer.

        :param b: The buffer to read into.
        :return: The number of bytes read into the buffer.
        """
        size = min(len(b), self._size - self._pos)
        b[:size] = cast('bytes', self._res[self._pos : self._pos + size])
        self._pos += size
        return size

    @property
    def closed(self) -> bool:
        """
        Check if the stream is closed.

        :return: True if the stream is closed, False otherwise.
        """
        return self._res.closed

    def close(self) -> None:
        """
        Close the stream and the underlying ResourceFile.
        """
        super().close()
        self._res.close()

    def partial(self, offset: int, size: int) -> IO[bytes]:
        """
        Get a partial view of the stream as a new stream.

        :param offset: The offset to start the partial view.
        :param size: The size of the partial view.
        :return: A new stream representing the partial view.
        """
        if size > io.DEFAULT_BUFFER_SIZE:
            res = ResourceFile(self._res[offset : offset + size])
            return io.BufferedReader(ResourceStream(res))
        return io.BytesIO(self._res[offset : offset + size])  # type: ignore[arg-type]


class ResourceFile(AbstractContextManager[memoryview]):
    """
    A class representing a resource file that can be memory-mapped.
    This class supports context management and provides a buffer interface.
    """

    __slots__ = ('_tmpfile', 'buffer', 'closed')

    def __init__(
        self,
        buffer: ArrayLike,
        tmpfile: 'IO[bytes] | None' = None,
    ) -> None:
        """
        Initialize the ResourceFile with a buffer and an optional temporary file.

        :param buffer: The buffer to wrap.
        :param tmpfile: An optional temporary file.
        """
        self.buffer = memoryview(buffer)  # type: ignore[arg-type]
        self.closed = False
        self._tmpfile = tmpfile

    def __len__(self) -> int:
        """
        Get the length of the buffer.

        :return: The length of the buffer.
        """
        if self.closed:
            raise ValueError('I/O operation on closed file')  # noqa: TRY003
        return len(self.buffer)

    def __buffer__(self, _flags: int) -> memoryview:
        """
        Get the buffer interface.

        :param _flags: Flags for the buffer interface.
        :return: The buffer interface.
        """
        if self.closed:
            raise ValueError('I/O operation on closed file')  # noqa: TRY003
        return self.buffer

    def __exit__(
        self,
        /,
        __exc_type: type[BaseException] | None,
        __exc_value: BaseException | None,
        __traceback: TracebackType | None,
    ) -> bool | None:
        """
        Exit the context manager and close the resource file.

        :param __exc_type: The exception type, if any.
        :param __exc_value: The exception value, if any.
        :param __traceback: The traceback, if any.
        :return: None.
        """
        self.close()
        return None

    @overload
    def __getitem__(self, index: slice) -> ArrayLike: ...
    @overload
    def __getitem__(self, index: int) -> int: ...
    def __getitem__(self, index: slice | int) -> ArrayLike | int:
        """
        Get an item or a slice from the buffer.

        :param index: The index or slice to get.
        :return: The item or slice from the buffer.
        :raises OSError: If the file is closed.
        """
        if not self.closed:
            return self.buffer[index]
        raise ValueError('I/O operation on closed file')  # noqa: TRY003

    @classmethod
    def load(
        cls,
        file_path: str | bytes | os.PathLike[AnyStr],
        key: int = 0x00,
        *,
        copy: bool = True,
    ) -> 'ResourceFile':
        """
        Load a file into a ResourceFile, optionally applying a key for XOR encryption.

        :param file_path: The path to the file to load.
        :param key: The key to use for XOR encryption (0x00 means no encryption).
        :return: The loaded ResourceFile.
        """

        file_size = os.path.getsize(file_path)  # noqa: PTH202

        if file_size < io.DEFAULT_BUFFER_SIZE**2:  # < 64MB
            data = np.fromfile(file_path, dtype='u1')
            if key == 0x00:
                return cls(data.tobytes())
            return cls((data ^ key).tobytes())

        data = np.memmap(file_path, dtype='u1', mode='r')
        if key == 0x00 and not copy:
            return cls(data)
        tmp = tempfile.TemporaryFile()  # noqa: SIM115
        # if no xor key is provided, copy the file into the tmp file
        if key == 0x00:
            data.tofile(tmp)
        else:
            result = np.memmap(tmp, dtype='u1', mode='w+', shape=data.shape)
            result[:] = data ^ key
            result.flush()
            del result
        return cls(np.memmap(tmp, dtype='u1', mode='r', shape=data.shape), tmp)

    def close(self) -> None:
        """
        Close the resource file and the underlying temporary file, if any.
        """
        if self._tmpfile is not None:
            self._tmpfile.close()
            self._tmpfile = None
        if hasattr(self, 'buffer'):
            del self.buffer
        self.closed = True

    def __del__(self) -> None:
        """
        Clean up resources when the object is garbage collected.
        """
        self.close()

    @classmethod
    def open(
        cls,
        file: str | bytes | os.PathLike[AnyStr],
        mode: str = 'r',
        encoding: str = 'utf-8',
        errors: str | None = None,
    ) -> AbstractContextManager[IO[AnyStr]]:
        """
        Open a file and return an iterator over the file's contents.

        :param file: The path to the file to open.
        :param mode: The mode to open the file in (default is 'r').
        :param encoding: The encoding to use (default is 'utf-8').
        :param errors: The error handling scheme (default is None).
        :return: An iterator over the file's contents.
        """
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
    """
    Read the contents of a file, optionally applying a key for XOR encryption.

    :param file_path: The path to the file to read.
    :param key: The key to use for XOR encryption
        (default is 0x00, which means no encryption).
    :return: The contents of the file as bytes.
    """
    with ResourceFile.load(file_path, key=key) as res:
        return bytes(res)
