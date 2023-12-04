# Contributing to Docugami

Hi there! Thank you for even being interested in contributing to Docugami's dgml-utils.
As an open-source project in a rapidly developing field, we are extremely open to contributions, whether they involve new features, improved infrastructure, better documentation, or bug fixes.

## 🗺️ Guidelines

### 👩‍💻 Contributing Code

To contribute to this project, please follow the ["fork and pull request"](https://docs.github.com/en/get-started/quickstart/contributing-to-projects) workflow.
Please do not try to push directly to this repo unless you are a maintainer.

Please follow the checked-in pull request template when opening pull requests. Note related issues and tag relevant
maintainers.

Pull requests cannot land without passing the formatting, linting, and testing checks first. See [Testing](#testing) and
[Formatting and Linting](#formatting-and-linting) for how to run these checks locally.

If there's something you'd like to add or change, opening a pull request is the
best way to get our attention.

### 🚩GitHub Issues

Our [issues](https://github.com/docugami/dgml-utils/issues) page is kept up to date with bugs, improvements, and feature requests.

If you start working on an issue, please assign it to yourself.

If you are adding an issue, please try to keep it focused on a single, modular bug/improvement/feature.
If two issues are related, or blocking, please link them rather than combining them.

We will try to keep these issues as up-to-date as possible, though
with the rapid rate of development in this field some may get out of date.
If you notice this happening, please let us know.

### 🙋Getting Help

Our goal is to have the simplest developer setup possible. Should you experience any difficulty getting setup, please
contact a maintainer! Not only do we want to help get you unblocked, but we also want to make sure that the process is
smooth for future contributors.

In a similar vein, we do enforce certain linting, formatting, and documentation standards in the codebase.
If you are finding these difficult (or even just annoying) to work with, feel free to contact a maintainer for help -
we do not want these to get in the way of getting good code into the codebase.

### Local Development Dependencies

Install dgml-utils development requirements (for running dgml-utils, running examples, linting, formatting, tests, and coverage):

```bash
poetry install
```

Then verify dependency installation:

```bash
make test
```

### Testing

Unit tests cover modular logic that does not require calls to outside APIs.
If you add new logic, please add a unit test.

To run unit tests:

```bash
make test
```

### Formatting and Linting

Run these locally before submitting a PR; the CI system will check also.

#### Code Formatting

Formatting for this project is done via [ruff](https://docs.astral.sh/ruff/rules/).

To run formatting for docs, cookbook and templates:

```bash
make format
```

#### Linting

Linting for this project is done via a combination of [ruff](https://docs.astral.sh/ruff/rules/) and [mypy](http://mypy-lang.org/).

To run linting for docs, cookbook and templates:

```bash
make lint
```

We recognize linting can be annoying - if you do not want to do it, please contact a project maintainer, and they can help you with it. We do not want this to be a blocker for good code getting contributed.

## 🏭 Release Process

As of now, Docugami has an ad hoc release process: releases are cut with high frequency by
a developer and published to [PyPI](https://pypi.org/project/dgml-utils/).
