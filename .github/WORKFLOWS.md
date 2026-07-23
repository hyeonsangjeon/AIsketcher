# Release and deployment workflows

Normal CI only validates the repository; it never publishes a package or
deploys documentation. Release and Pages workflows stay separate from pull
request validation and use protected deployment boundaries.

- `ci.yml` tests the core SDK, builds documentation strictly, and inspects
  distribution artifacts.
- `pages.yml` is manual-only and deploys the already reviewable MkDocs site to
  GitHub Pages. Merging to `main` or publishing a package does not deploy the
  documentation site automatically.
- `publish-pypi.yml` starts when a GitHub Release is published from a matching
  `v<version>` tag. It derives and verifies the package version, builds and
  smoke-tests the distributions in a read-only job, attaches those verified
  files to the GitHub Release, and sends the same artifact to the protected
  `pypi` environment using Trusted Publishing. A confirmed manual dispatch from
  the exact tag remains available for recovery after a failed publication. The
  recovery path requires an existing, non-draft GitHub Release for that exact
  tag and attaches the same verified distributions after PyPI succeeds. If
  PyPI already contains both files with the exact same SHA-256 digests, a retry
  skips the immutable upload and continues to Release-asset recovery; a missing
  or different file fails closed.

`README.md` is the project description embedded in the built wheel and source
archive. Publishing a new tagged release therefore updates the PyPI page with
the README from that immutable artifact. A later README edit on `main`, or an
edit to an existing GitHub Release, does not rewrite an already-published PyPI
version.

The `release.published` event must come from a maintainer, GitHub App, or user
token. GitHub intentionally does not start a second workflow when another
workflow creates the Release with its default `GITHUB_TOKEN`; use the confirmed
tag dispatch in that case. Release assets are attached only after PyPI accepts
the verified wheel and source archive. A retry keeps an existing asset only
when its bytes match the verified distribution, uploads missing assets, and
fails instead of replacing an asset with different bytes.

Workflow permissions are read-only unless a deployment job needs Pages, release
assets, or OIDC. Every third-party action is pinned to the full commit resolved
from its reviewed upstream release tag on 2026-07-24; the readable version
remains beside the SHA as a YAML comment. Dependabot may propose a newer
version, but the reviewed update must preserve full-SHA pinning.

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
publishing a GitHub Release. Do not add an API token as a repository secret.
