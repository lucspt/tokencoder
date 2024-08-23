# Tokencoder

Hey, welcome to `tokencoder`! It's a tokenizer package that wraps [`tiktoken`](https://github.com/openai/tiktoken/tree/main),
and also allows you to train your own tokenizers from scratch.

[`tiktoken`](https://github.com/openai/tiktoken/tree/main) is a very fast BPE tokenizer package. By default it is
for use with OpenAI models, therefore you can't train your own tokenizers. That is the problem `tokencoder` solves.

Contents:

- [Training a tokenizer](#training-your-own-tokenizer)
- [Constructing a tokenizer](#constructing-a-tokenizer)

## Training your own tokenizer

To train your own tokenizer is very simple:

```python
from tokencoder.trainer import TokenizerTrainer
from tokencoder.patterns import DEFAULT_REGEX_PATTERN

trainer = TokenizerTrainer(
  name="your-name-here",
  special_tokens={
    "a",
    "set",
    "of",
    "special_tokens",
  },
  regex_pattern_string=DEFAULT_REGEX_PATTERN,  # Optional. This is the default behavior
)

tokenizer_path = trainer.train(
  text="your text corpus",
  vocab_size=257, # must be greater than 256
  save_dir="path/to/save/your/tokenizer" # optional, defaults to cwd
)
```

The example above writes a json file to `tokenizer_path` with all the data you need to construct
a `Tokenizer` class.

## Constructing a tokenizer

Once you have trained and saved your tokenizer to a file, you can use the `from_file` method
to load it.

```python
from tokencoder import Tokenizer

tokenizer = Tokenizer.from_file("/path/to/tokenizer")
text = "hello tokencode!"
assert tokenizer.decode(tokenizer.encode(text)) == text
```

The `Tokenizer` class is a subclass of `tiktoken`'s `Encoding` class, meaning that
it behaves exactly like a tiktoken encoding would.

Also, the `from_file` method accepts all arguments that the `Encoding` init method does,
as keyword arguments. See [here](https://github.com/openai/tiktoken/tree/main?tab=readme-ov-file#extending-tiktoken)
for more

## Wrapping up

That's about it, have fun!
