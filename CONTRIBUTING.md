# Contributing

We are excited for you to start contributing! Here are some guidelines:

- [Find A Bug?](#find-a-bug)
- [Development Setup](#development-setup)

## Find A Bug?

If you find a bug in the code, you can submit a pull request with a fix.

## Development Setup

We use [Rye](https://rye.astral.sh/guide/) to manage dependencies, if you do not know it, it's great! [Check it out](https://rye.astral.sh/guide/installation/)!

After you install it, you just have to run:

```bash
rye sync --all-features && source ./scripts/prepare
```

You can then run scripts using `rye run python script.py`.

### Commonly used rye scripts

```bash
# run ruff linter and formatter
rye run fix

# run mypy type checker and ruff linter
rye run lint

# run test w/ coverage report
rye run test
# OR for html report
rye run test-ui
```

And that's about all you need to know to start contributing!
