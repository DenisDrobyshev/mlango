# Governance

Short, because the project is small and pretending otherwise would help nobody.

## Who decides

mlango has one maintainer: [Denis Drobyshev](https://github.com/DrobyshevDev).
Design decisions, releases and what goes into the framework are his call.

That is a fact about the project's size, not a preference. As contributors
accumulate a track record, commit access follows, and this file gets longer.

## How decisions get made

Anything that changes what a user writes — a new `Meta` option, a change to a
contract, a new declarative family — starts as a GitHub issue before it starts
as a pull request. Not for ceremony: the framework's whole argument is that
there is one obvious way to declare a thing, and a second obvious way arriving
by pull request is the cheapest possible time to notice.

Bug fixes, documentation, error messages and tests need no issue. Open the pull
request.

Three questions decide whether a feature belongs in the framework rather than in
a package:

1. Would most projects want it, or would most projects have to work around it?
2. Can it be read from `_meta` rather than special-cased per family?
3. Does it add a dependency that everyone then installs?

A "no" to the first, or a "yes" to the third, is usually an argument for
[an extension package](docs/extending.md) — which needs nobody's approval and
is found automatically once installed. The framework staying installable without
a compiler is worth more than a longer list of built-ins.

## What gets rejected

- Features that add a second way to do something that already has one.
- Layering violations: `core` importing from anywhere, or the four declarative
  families importing each other. See the table in [CONTRIBUTING.md](CONTRIBUTING.md).
- Anything that makes the first five minutes worse. `startproject` must keep
  producing a project that runs.
- Code without tests, or with types that do not check.

None of these are personal, and all of them are negotiable if the reasoning is
good. Say why in the issue.

## Releases

Semantic versioning. Before 1.0, minor versions may change contracts, and the
changelog says so explicitly when they do. `CHANGELOG.md` is written as the work
lands, not assembled at release time. See [RELEASING.md](RELEASING.md).

## If the maintainer disappears

The project is MIT-licensed and everything needed to continue it is in the
repository: the tests, both CI pipelines, the release workflow and the
documentation source. Fork it. That is not a failure mode this file needs to
prevent, only one it should name.

## Code of conduct

Participation is covered by the [Code of Conduct](CODE_OF_CONDUCT.md). It
applies to issues, pull requests and any other project space, and the maintainer
enforces it.
