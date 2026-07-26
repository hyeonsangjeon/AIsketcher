# Release and deployment workflows

Normal CI only validates the repository; it never publishes a package or
deploys documentation. Release and Pages workflows stay separate from pull
request validation and use protected deployment boundaries.

- `ci.yml` tests the core SDK, runs the model-free Playwright Studio regression
  suite, builds documentation strictly, and inspects distribution artifacts.
- `pages.yml` is manual-only and deploys the already reviewable MkDocs site to
  GitHub Pages. Merging to `main` or publishing a package does not deploy the
  documentation site automatically.
- `publish-pypi.yml` starts when a GitHub Release is published from a matching
  `v<version>` tag. It derives and verifies the package version, requires a
  successful `ci.yml` run for that exact immutable event SHA, builds and
  smoke-tests the distributions once in a read-only job, and records both
  SHA-256 digests. That Actions artifact enters the protected `pypi` environment
  for Trusted Publishing. A confirmed manual dispatch from the current default
  branch can select an exact existing tag for recovery after a failed or partial
  publication. This deliberately uses the repaired workflow from `main` while
  building the immutable tagged source; rerunning an old release run would reuse
  the old workflow definition. The recovery path requires an existing,
  non-draft GitHub Release for that exact tag. It
  downloads every file PyPI already has, verifies its advertised SHA-256, and
  compares its normalized archive contents with the newly verified build. A
  pre-existing wheel must also match the recorded Actions wheel SHA-256 and bytes
  exactly. A pre-existing source archive may differ only in archive-container
  representation that the normalized equivalence verifier intentionally
  excludes, such as timestamps and compression metadata. An
  unexpected or content-different file fails closed; a valid strict subset
  stages and publishes only the missing distribution. The completed wheel and
  source archive are then downloaded from PyPI, verified again, preserved as the
  canonical PyPI/Release distribution set, and attached to the GitHub Release
  without replacing different existing assets.

Publishing the GitHub Release is the human release gate. The `pypi` GitHub
Environment does not require a second environment approval, so a
`release.published` run continues automatically only after the immutable tag,
default-branch ancestry, exact-SHA CI, test, build, scan, smoke-test, and digest
checks pass. The environment keeps the Trusted Publisher identity and limits
normal deployments to tags matching `v*`; the branch `main` is allowed only for
the explicitly confirmed recovery dispatch. The publish job rechecks the
downloaded Actions artifact against the build outputs, uploads only
distributions that PyPI does not already have, and downloads the complete PyPI
wheel and source archive. The wheel must always match the recorded Actions
SHA-256 and downloaded bytes exactly, whether it was already present or newly
uploaded. Every other newly uploaded file is also exact. A pre-existing source
archive is the sole recovery exception: it must match its PyPI digest and the
normalized recorded archive contents, but its container representation may
differ.

The build also derives `SOURCE_DATE_EPOCH` from the immutable tag commit before
building the wheel directly from the reviewed source, separately from the
source archive. The direct same-toolchain wheel build is byte-reproducible and
supports the exact-wheel recovery check. Setuptools source archives are not
byte-reproducible in this configuration even with that timestamp, so normalized
equivalence is permitted only for an already published source archive. The
initial Actions artifact remains the canonical identity throughout automatic
publication and recovery.

`README.md` is the project description embedded in the built wheel and source
archive. Publishing a new tagged release therefore updates the PyPI page with
the README from that immutable artifact. A later README edit on `main`, or an
edit to an existing GitHub Release, does not rewrite an already-published PyPI
version.

The standard release path requires a maintainer to publish the GitHub Release.
If a GitHub App or another workflow is later allowed to publish Releases
automatically, restore a required environment reviewer or an equivalent human
gate first. Before any checkout, build, or PyPI upload, the workflow requires an
existing published, non-prerelease GitHub Release whose tag resolves to a commit
in the current default-branch history. A release event must still match its
immutable event SHA; a manual recovery resolves and records the immutable tag
SHA independently of the workflow's `main` SHA. It also waits for a successful
first-party CI run whose head SHA is that same tag commit; a successful run for
another commit cannot authorize the release. The same Release, tag,
resolved-SHA, and ancestry checks run again immediately before the PyPI state
check and OIDC upload. GitHub intentionally does not start a second workflow
when another workflow creates the Release with its default `GITHUB_TOKEN`; use
the confirmed default-branch dispatch only after that Release exists. Release
assets are attached only after PyPI exposes and verifies the complete wheel and
source-archive set. A retry keeps an existing asset only when its bytes match
the canonical PyPI distribution, uploads missing assets, and fails instead of
replacing an asset with different bytes.

Workflow permissions are read-only unless a deployment job needs Pages, release
assets, or OIDC. The release build has read-only Actions access only to verify
the exact-SHA CI result, and the Pages build has read-only Pages access only to
resolve the configured site. Every third-party action is pinned to the full
commit resolved from its reviewed upstream release tag on 2026-07-24; the
readable version remains beside the SHA as a YAML comment. Dependabot may
propose a newer version, but the reviewed update must preserve full-SHA pinning.

Upstream release pages:

- <https://github.com/actions/checkout/releases>
- <https://github.com/actions/setup-python/releases>
- <https://github.com/actions/upload-artifact/releases>
- <https://github.com/actions/download-artifact/releases>
- <https://github.com/actions/configure-pages/releases>
- <https://github.com/actions/upload-pages-artifact/releases>
- <https://github.com/actions/deploy-pages/releases>
- <https://github.com/pypa/gh-action-pypi-publish/releases>

Configure PyPI Trusted Publishing for owner `hyeonsangjeon`, repository
`AIsketcher`, workflow `publish-pypi.yml`, and environment `pypi` before
publishing a GitHub Release. Configure that environment with selected
deployment refs only: branch `main` and tags matching `v*`. Do not configure a
required reviewer unless a second manual hold point is intentionally desired;
otherwise it recreates the `Review pending deployments` pause after every
Release. If the repository's default branch is renamed, update the literal
`main` environment policy at the same time. Do not add an API token as a
repository secret.
