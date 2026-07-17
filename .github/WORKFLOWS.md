# Release and deployment workflows

Normal CI only validates the repository; it never publishes a package or
deploys documentation. Release and Pages workflows stay separate from pull
request validation and use protected deployment boundaries.

- `ci.yml` tests the core SDK, builds documentation strictly, and inspects
  distribution artifacts.
- `pages.yml` is manual-only and deploys the already reviewable MkDocs site to
  GitHub Pages.
- `publish-pypi.yml` starts when a GitHub Release is published from a matching
  `v<version>` tag. It derives and verifies the package version, builds and
  smoke-tests the distributions in a read-only job, attaches those verified
  files to the GitHub Release, and sends the same artifact to the protected
  `pypi` environment using Trusted Publishing. A confirmed manual dispatch from
  the exact tag remains available for recovery after a failed publication.

The `release.published` event must come from a maintainer, GitHub App, or user
token. GitHub intentionally does not start a second workflow when another
workflow creates the Release with its default `GITHUB_TOKEN`; use the confirmed
tag dispatch in that case. Release assets are attached only after PyPI accepts
the verified wheel and source archive, and existing assets are never
overwritten by a retry.

Workflow permissions are read-only unless a deployment job needs Pages or OIDC.
Actions use the official current major tags verified from their upstream
release pages on 2026-07-16; the PyPI publisher uses its exact v1.14.0 release
tag. Commit SHAs were not guessed during scaffolding. Before enabling protected
production environments, review each upstream release and pin the approved
action to its full commit SHA through a dependency update.

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
