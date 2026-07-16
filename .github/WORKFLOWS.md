# Release and deployment workflows

Normal CI only validates the repository; it never publishes a package or
deploys documentation. Release and Pages workflows are separate, manual, and
review-gated.

- `ci.yml` tests the core SDK, builds documentation strictly, and inspects
  distribution artifacts.
- `pages.yml` is manual-only and deploys the already reviewable MkDocs site to
  GitHub Pages.
- `publish-pypi.yml` is manual-only, requires an exact version confirmation,
  requires dispatching the matching `v<version>` tag, and targets the protected
  `pypi` GitHub environment using Trusted Publishing.

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
- <https://github.com/actions/configure-pages/releases>
- <https://github.com/actions/upload-pages-artifact/releases>
- <https://github.com/actions/deploy-pages/releases>
- <https://github.com/pypa/gh-action-pypi-publish/releases>

Configure PyPI Trusted Publishing for this repository, workflow filename, and
the `pypi` environment before running the package workflow. Do not add an API
token as a repository secret.
