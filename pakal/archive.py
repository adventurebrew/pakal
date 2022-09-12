import io
import os
import pathlib
from contextlib import AbstractContextManager, contextmanager
from types import TracebackType
from typing import (
    IO,
    Any,
    AnyStr,
    Callable,
    ContextManager,
    Generic,
    Iterator,
    Mapping,
    NamedTuple,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

from pakal.stream import PartialStreamView


GLOB_ALL = '*'


ArchiveType = TypeVar('ArchiveType', bound='BaseArchive')
EntryType = TypeVar('EntryType')
ArchiveIndex = Mapping[str, EntryType]


def create_directory(name: AnyStr) -> None:
    os.makedirs(name, exist_ok=True)


class _SimpleEntry(NamedTuple):
    offset: int
    size: int


SimpleEntry = Union[_SimpleEntry, Tuple[int, int]]


def read_file(stream: IO[bytes], offset: int, size: int) -> IO[bytes]:
    stream.seek(
        offset, io.SEEK_SET
    )  # need unit test to check offset is always equal to f.tell()
    return cast(IO[bytes], PartialStreamView(stream, size))


class ArchivePath:
    def __init__(
        self,
        fname: Union[str, os.PathLike[str]],
        archive: 'BaseArchive[EntryType]',
    ) -> None:
        self.fname = os.path.normpath(fname)
        self.archive = archive

    @property
    def parent(self) -> 'ArchivePath':
        return ArchivePath(pathlib.Path(self.fname).parent, self.archive)

    @property
    def name(self) -> str:
        return str(pathlib.Path(self.fname).name)

    def __str__(self):
        return str(self.fname)

    def match(self, pattern: str):
        return pathlib.Path(self.fname).match(pattern)

    def exists(self) -> bool:
        return any(self.archive.glob(str(self)))

    def glob(self, pattern: str) -> Iterator['ArchivePath']:
        return (
            entry
            for entry in self.archive
            if entry.match(os.path.join(self.fname, pattern))
        )

    def open(
        self,
        mode: str = 'r',
        encoding: str = 'utf-8',
        errors: Optional[str] = None,
    ) -> ContextManager[IO[AnyStr]]:
        return self.archive.open(
            self.fname,
            mode=mode,
            encoding=encoding,
            errors=errors,
        )

    def read_bytes(self):
        with self.open(mode='rb') as f:
            return f.read()

    def read_text(self, encoding=None, errors=None):
        with self.open(
            mode='r',
            encoding=encoding,
            errors=errors,
        ) as f:
            return f.read()

    def __truediv__(self, key: Union[str, os.PathLike[str]]) -> 'ArchivePath':
        return ArchivePath(str(pathlib.Path(self.fname) / key), self.archive)

    def __rtruediv__(self, key: Union[str, os.PathLike[str]]) -> 'ArchivePath':
        return ArchivePath(str(key / pathlib.Path(self.fname)), self.archive)


class BaseArchive(AbstractContextManager, Generic[EntryType]):
    _stream: IO[bytes]

    index: Mapping[str, EntryType]

    def _create_index(self) -> ArchiveIndex[EntryType]:
        raise NotImplementedError('create_index')

    @contextmanager
    def _read_entry(self, entry: EntryType) -> Iterator[IO[bytes]]:
        raise NotImplementedError('read_entry')

    def __init__(self, file: Union[AnyStr, os.PathLike[AnyStr], IO[bytes]]) -> None:
        if isinstance(file, os.PathLike):
            file = os.fspath(file)

        if isinstance(file, (str, bytes)):
            self._stream = io.open(file, 'rb')
        else:
            self._stream = file
        self.index = {
            os.path.normpath(name): entry
            for name, entry in self._create_index().items()
        }

    @contextmanager
    def open(
        self,
        fname: str,
        mode: str = 'r',
        encoding: str = 'utf-8',
        errors: Optional[str] = None,
    ) -> Iterator[IO[AnyStr]]:
        try:
            member = self.index[os.path.normpath(fname)]
        except KeyError:
            raise ValueError(f'no member {fname} found in archive')

        stream: IO
        with self._read_entry(member) as stream:
            if 'b' not in mode:
                stream = io.TextIOWrapper(
                    cast(IO[bytes], stream),
                    encoding=encoding,
                    errors=errors,
                )
            yield stream

    def close(self) -> Optional[bool]:
        return self._stream.close()

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        return self.close()

    def __iter__(self) -> Iterator[ArchivePath]:
        for fname, _ in self.index.items():
            yield ArchivePath(fname, self)

    def glob(self, pattern: str) -> Iterator[ArchivePath]:
        return (entry for entry in self if entry.match(pattern))

    def extractall(
        self,
        dirname: Union[str, os.PathLike[str]],
        pattern: str = GLOB_ALL,
    ) -> None:
        dirname = pathlib.Path(dirname)
        for entry in self.glob(pattern):
            os.makedirs(str(dirname / entry.parent), exist_ok=True)
            with io.open(str(dirname / entry.fname), 'wb') as out_file:
                out_file.write(entry.read_bytes())


class SimpleArchive(BaseArchive[SimpleEntry]):
    @contextmanager
    def _read_entry(self, entry: SimpleEntry) -> Iterator[IO[bytes]]:
        entry = _SimpleEntry(*entry)
        yield read_file(self._stream, entry.offset, entry.size)


def make_opener(
    archive_type: Type[ArchiveType],
) -> Callable[..., ContextManager[ArchiveType]]:
    @contextmanager
    def opener(*args: Any, **kwargs: Any) -> Iterator[ArchiveType]:
        with archive_type(*args, **kwargs) as inst:
            yield inst

    return opener


if __name__ == '__main__':
    import argparse
    from importlib import import_module

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument(
        '-m', '--module', required=True, type=str, help='Archive module'
    )
    parser.add_argument('filename', type=str, help='File to extract from')
    parser.add_argument(
        'pattern',
        type=str,
        nargs='?',
        default=GLOB_ALL,
        help='Pattern of file names to extract',
    )

    args = parser.parse_args()

    open_archive = getattr(import_module(args.module), 'open')

    print(args.filename, args.pattern)
    with open_archive(args.filename) as arc:

        if args.pattern == GLOB_ALL:
            assert set(str(x) for x in arc.glob(args.pattern)) == set(
                fname for fname in arc.index
            )

        arc.extractall('out', args.pattern)
