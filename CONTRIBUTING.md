# Contributing to Tesseract-JAX

Tesseract-JAX is an open-source project and, as such, we welcome contributions
from developers, engineers, scientists, and end-users in general. Contributions
are what make the open source community such an amazing place to learn,
inspire, and create. Any contributions you make are greatly appreciated.

## Code of Conduct

Ensure your contributions adhere to the [Code of Conduct](CODE_OF_CONDUCT.md).

## Feedback

Constructive feedback is very welcome. We are interested in hearing from you!

In the case things aren't working as expected, or the documentation is lacking,
please [file a bug
report](https://github.com/pasteurlabs/tesseract-jax/issues/new?template=BUG-REPORT.yml).

In the case you want to suggest a new feature, please file a new [feature
request](https://github.com/pasteurlabs/tesseract-jax/issues/new?template=FEATURE-REQUEST.yml).
In particular, we recommend you open an issue before contributing code in a
pull request. This allows all parties to talk things over before jumping into
action, and increase the likelihood of pull requests getting merged.

In case you have general questions or feedback, need support from the
community, or have a cool demo to share, start a thread in our [Discourse
Forum](https://si-tesseract.discourse.group/). We use GitHub Issues for bug
reports and feature requests only.

## Documentation

Tesseract-JAX documentation is kept under the `docs/` directory of the repository,
written in Markdown and using Sphinx to generate the final HTMLs. Fixes and
enhancements to the documentation should be submitted as pull requests, we
treat the same as code contributions.

To build the documentation locally, install the documentation dependencies in
addition to the project itself, then run `make html`:

```console
$ . venv/bin/activate
$ pip install -e .[dev]
$ pip install -r docs/requirements.txt
$ cd docs
$ make html
```

The resulting HTMLs are in `docs/build/html/`.

Contributions in the form of tutorials, examples, demos, blog posts (including
those posted elsewhere already) are best highlighted and celebrated in the
[Discourse Forum](https://si-tesseract.discourse.group/).

## Code

Tesseract-JAX is developed under the [Apache 2.0](LICENSE) license. By contributing
to the Tesseract-JAX project you agree that your code contributions are governed by
this license. We require you to sign our [Contributor License
Agreement](https://github.com/pasteurlabs/pasteur-oss-cla/blob/main/README.md)
to state so.

### Local development setup

Make sure you have [Docker installed](https://docs.docker.com/engine/install/)
on your machine and you can run `docker` commands via your user. After that,
clone the repository, install the dependencies, and setup pre-commit hooks:

```console
$ git clone git@github.com:pasteurlabs/tesseract-jax.git
$ cd tesseract-jax
$ python -m venv venv
$ . venv/bin/activate
$ pip install -e .[dev]
$ pre-commit install
```

### Building the GPU-direct shim from source

Tesseract-JAX ships an optional native shim (`tesseract_jax/_cuda_shim.cc`) that
enables the GPU-direct (`cuda_ipc`) transport, exchanging device arrays with a
served Tesseract without a host round-trip. The published wheels bundle it, but a
source install (`pip install -e .`) has to compile it. The shim needs:

- A C++17 compiler (`c++` by default; override with the `CXX` environment
  variable). It links no CUDA library (the CUDA runtime is `dlopen`ed at import),
  so no CUDA toolkit is required at build time.
- `nanobind` and `jaxlib` (for the XLA FFI headers), which are declared as
  build-time dependencies and installed automatically under build isolation. The
  shim is compiled against the _oldest_ supported `jaxlib` on purpose (pinned in
  `pyproject.toml`'s `[build-system].requires`). XLA's FFI ABI guarantee only
  covers running an old-jaxlib shim against a newer runtime, so building at the
  floor keeps one shim valid across the whole supported `jaxlib` range.

The build **degrades gracefully by default**: if the compiler or headers are
missing, the shim is skipped, the package still installs and imports, and the
GPU-direct path reports itself unavailable (`tesseract_jax.gpu_ffi.is_available()`
returns `False`) so callers fall back to the host-callback transport. Two
environment variables control this:

- `TESSERACT_JAX_GPU_REQUIRED=1` makes a shim build failure **fatal** instead of
  falling back. Use it when you need to be sure the shim actually built (CI sets
  it for the GPU jobs).
- `TESSERACT_JAX_PURE_PYTHON=1` **skips** the shim entirely and produces a
  pure-Python install, even on a platform that could compile it.

If you install with `--no-build-isolation`, pre-install the build-time
dependencies first (see `pyproject.toml`'s `[build-system].requires` for the
exact `jaxlib` pin per Python version):

```console
$ pip install "hatchling" "hatch-vcs" "setuptools-scm!=9.0.0" "nanobind>=2.0" "jaxlib"
$ pip install -e . --no-build-isolation
```

The version is derived from Git tags via `hatch-vcs`. A shallow clone without
tags falls back to `0.0.0+unknown`, so fetch tags (`git fetch --tags`) before
building if you care about the reported version.

### Tests

This project uses the pytest framework for all tests. New code should be
covered by new or existing tests.

To run the tests simply run `pytest` in the root of the project:

```console
$ pytest
```

The GPU-direct tests are marked with the `gpu` marker and require a CUDA device
plus a shim build (see above); they are skipped automatically when no GPU is
available. CI runs them across a small matrix of CUDA majors (12 and 13), Python
versions (the 3.12 floor and 3.14 ceiling), and JAX versions (latest, plus the
declared `jax==0.7.0` floor on the CUDA 12 leg) so the shim's runtime loader and
the `cuda_ipc` path stay covered across the support window.

Behaviours that must match between the host and GPU-direct transports (dtype
handling, discarded-slot fills, non-differentiable inputs/outputs, jacobian
fwd/bwd, batching) are written once in `tests/test_transport_parity.py` and run on
both via the parametrised `transport` fixture, which serves the array-agnostic
`tests/transport_tesseract` with `numpy` or `cupy` compute to match. Tests that
have no host analogue (FFI-boundary fault injection, on-device residency checks)
stay in `tests/test_gpu_direct.py`.

### GitHub workflow

This project uses Git for version control and follows a GitHub workflow. To
contribute follow these steps:

1. Fork the project via the GitHub UI.
1. Clone your fork to your machine.
1. Add an upstream remote: `git remote add upstream git@github.com:pasteurlabs/tesseract-jax.git`.
1. Create a new branch for your code contribution: `git switch --create my_branch`.
1. Implement your changes.
1. Commit and push to your fork: `git push --set-upstream origin my_branch`.
1. [Open a Pull Request](https://github.com/pasteurlabs/tesseract-jax/pulls) with
   your changes.

It is a good practice to rebase often on top of `main` to keep your code up to
date with latest development and minimize merge conflicts:

```console
$ git fetch upstream
$ git switch main
$ git merge upstream/main
$ git switch my_branch
$ git rebase main
$ git push --force
```

### Commit and pull request messages guidelines

We follow the [Conventional
Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification for all
commits that reach the `main` branch. Each commit is crafted from a pull
request that is squash-merged. The commit title and message comes from the pull
request title and message, respectively. As such, they should be structured
following the specfication.

The title consists of a _type_, and optional _scope_, and a short
_description_: `type[(scope)]: description`. The types we use are:

- `chore`: for changes that affect the build system, external dependencies, or
  general housekeeping.
- `ci`: for changes in the CI.
- `doc`: for documentation only changes.
- `feat`: for a new feature.
- `fix`: for fixing a bug.
- `perf`: for a code change that improves performance.
- `refactor`: for a code change that neither adds a feature nor fixes a bug.
- `security`: for a change that fixes a security issue.
- `test`: for adding new tests or fixing existing ones.

The scopes we use are:

- `cli`: for changes that affect CLI.
- `engine`: for changes that affect the CLI engine.
- `sdk`: for changes that affect the Python API.
- `example`: for changes in the examples.
- `runtime`: for changes in the runtime.
- `deps`: for changes in the dependencies.

In case there are breaking changes in your code, this should be indicated in
the message either by appending an exclamation mark (`!`) after the type/scope
or by adding a `BREAKING CHANGE:` trailer to the message.

## Versioning

The Tesseract-JAX project follows [semantic versioning](https://semver.org).

## Release process

(code owners only)

Releases are done via GitHub Actions, which automatically build the release
artifacts and publish them to the [GitHub Releases](https://github.com/pasteurlabs/tesseract-jax/releases) page. To create a new release, follow these steps:

1. Make sure the code is in a good state, all tests pass, and the documentation is up to date.
2. Trigger a new release action through the [GitHub UI](https://github.com/pasteurlabs/tesseract-jax/actions/workflows/release.yml). This opens a new pull request with the release notes and the version number.
3. Add any additional release notes to the pull request message. They will automatically be included at the top of the release notes.
4. In the meantime, you can add more commits to `main` (and update the release branch) which will trigger re-generation of the changelog and release notes.
5. Once the pull request is ready, merge it into `main`.
6. GitHub Actions will then automatically release the new version. Verify that the release artifacts are correctly built and published.
7. Make an announcement in the [Discourse Forum](https://si-tesseract.discourse.group/) and on social media, if applicable.
