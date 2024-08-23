from typing import Any, Callable, Iterable, Generator
from pathlib import Path

import pytest

from tokencoder.trainer import TokenizerTrainer
from tokencoder.patterns import DEFAULT_REGEX_PATTERN


@pytest.fixture
def train_tokenizer(
    tmp_path: Path,
) -> Generator[Callable[..., Path], None, None]:
    def _train(
        *,
        name: str,
        special_tokens: Iterable[str] = {},
        regex_pattern_string: str = DEFAULT_REGEX_PATTERN,
        **train_kwargs: Any,
    ) -> Path:
        trainer = TokenizerTrainer(
            name=name,
            special_tokens=special_tokens,
            regex_pattern_string=regex_pattern_string,
        )
        args: dict[str, Any] = {
            "text": Path(__file__).read_text() * 10,
            "vocab_size": 2**8 + 1,
            "save_dir": tmp_path,
        } | train_kwargs
        return Path(trainer.train(**args))

    yield _train
