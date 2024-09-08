import json
from typing import Callable
from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from tokencoder.trainer import TokenizerTrainer
from tokencoder.patterns import GPT2_REGEX_PATTERN, DEFAULT_REGEX_PATTERN

VOCAB_SIZE = 2**8 + 10


@pytest.fixture(scope="module")
def train_text() -> str:
    return Path(__file__).read_text()


class TestTokenizerTrainer:
    def test_count_pairs(self) -> None:
        pairs = [5, 5, 4, 4, 4]
        counts = TokenizerTrainer.count_pairs(pairs)
        assert counts[4, 4] == 2
        assert counts[5, 5] == 1

    def test_count_pairs_with_initial_counts(self) -> None:
        pairs = [4, 4, 4]
        counts = TokenizerTrainer.count_pairs(pairs, {(4, 4): 2})
        assert counts[4, 4] == 4

    def test_merge(self) -> None:
        idx, pair = -1, (1, 1)
        to_merge = [*pair, 2, *pair, 2]
        merged = TokenizerTrainer.merge(to_merge, pair=pair, idx=idx)
        assert merged == [idx, 2, idx, 2]

    def test_train(self, train_text: str, train_tokenizer: Callable[..., Path]) -> None:
        res = train_tokenizer(
            name="testing",
            text=train_text,
            vocab_size=VOCAB_SIZE,
        )
        assert res.exists() == True
        txt = res.read_text()
        assert len(txt) > 0
        assert res.stat().st_size > 0

    @pytest.mark.parametrize(
        "name,special_tokens,regex_pattern_string",
        [
            ("testone", {"<|endoftext|>"}, DEFAULT_REGEX_PATTERN),
            ("testtwo", {}, GPT2_REGEX_PATTERN),
        ],
    )
    def test_filepath_contains_tokenizer_data(
        self,
        name: str,
        special_tokens: set[str],
        train_tokenizer: Callable[..., Path],
        regex_pattern_string: str,
    ) -> None:
        pth = train_tokenizer(
            name=name,
            special_tokens=special_tokens,
            regex_pattern_string=regex_pattern_string,
        )

        with open(pth, "r") as f:
            data = json.load(f)

        for x in ["vocab", "regex_pattern", "name"]:
            assert x in data

        assert data["name"] == name
        assert data["regex_pattern"] == regex_pattern_string

        if special_tokens:
            assert "special_tokens" in data
            assert special_tokens & data["special_tokens"].keys()

    def test_train_when_file_exists(self, tmp_path: Path) -> None:
        name = "tokenizer_exists"
        pth = tmp_path / f"{name}.json"
        pth.touch()
        trainer = TokenizerTrainer(name="tokenizer_exists")
        with pytest.raises(FileExistsError):
            trainer.train(vocab_size=VOCAB_SIZE, text="hello world", save_dir=tmp_path)

    def test_warning_when_text_is_too_short_for_vocab_size(
        self, train_tokenizer: Callable[..., Path]
    ) -> None:
        with pytest.warns(UserWarning):
            train_tokenizer(text="abcd", vocab_size=1000, name="testing")

    def test_raises_when_vocab_size_is_lt_256(
        self, train_tokenizer: Callable[..., Path]
    ) -> None:
        with pytest.raises(ValueError):
            train_tokenizer(vocab_size=2**8 - 1, name="testing")

    def test_files_and_text_train_returns_same_merges(
        self, train_tokenizer: Callable[..., Path], train_text: str, tmp_path: Path
    ) -> None:
        vocab_size, spt = 300, {"<|endoftext|>"}
        text_pth = train_tokenizer(
            text=train_text, vocab_size=vocab_size, special_tokens=spt, name="text_test"
        )
        file_pth = train_tokenizer(
            files=[Path(__file__)],
            name="file_test",
            special_tokens=spt,
            vocab_size=vocab_size,
            file_read_chunksize=-1,
        )  # train text is this file
        files_pth = train_tokenizer(
            files=[Path(__file__), Path(__file__)],
            name="files_text",
            special_tokens=spt,
            vocab_size=vocab_size,
            file_read_chunksize=-1,
        )

        tknzrs = []
        for t in [text_pth, files_pth, file_pth]:
            with open(t, "r") as f:
                data = json.load(f)
                del data["name"]
                tknzrs.append(data)

        assert tknzrs[0] == tknzrs[1] == tknzrs[2]

    def test_raises_when_text_or_files_not_specified(
        self, train_tokenizer: Callable[..., Path]
    ) -> None:
        with pytest.raises(ValueError):
            train_tokenizer(vocab_size=300, name="rand", text=None, files=None)

    def test_chunksize(
        self, train_tokenizer: Callable[..., Path], mocker: MockerFixture
    ) -> None:
        from tokencoder.trainer import TokenizerTrainer as spy_class

        spy = mocker.spy(spy_class, "read_file_chunks")
        files, cs = [Path(__file__)], 1000000
        train_tokenizer(name="chunksize_test", files=files, file_read_chunksize=cs)
        spy.assert_called_once_with(files[0], chunksize=cs)

    @pytest.fixture
    def train_files(self, tmp_path: Path) -> list[Path]:
        files = [tmp_path / "1.txt", tmp_path / "2.txt"]
        for f in files:
            f.write_text("hello" * 100)
        return files

    def test_accepts_list_of_chunksizes(
        self,
        train_tokenizer: Callable[..., Path],
        tmp_path: Path,
    ) -> None:
        chunksizes = [-1, None]
        files = [tmp_path / "1.txt", tmp_path / "2.txt"]
        for f in files:
            f.write_text("hello" * 100)
        train_tokenizer(
            name="list_chunksizes",
            files=files,
            file_read_chunksize=chunksizes,
        )

    def test_read_file_chunks(self, train_files: list[Path]) -> None:
        trainer = TokenizerTrainer(name="test")
        chunks = trainer.files_to_chunks(files=train_files)
        chars: list[str] = []
        for x in chunks:
            chars.extend(chr(o) for o in x)
        assert "".join(chars) == "".join(x.read_text() for x in train_files)
