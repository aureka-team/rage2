---
type: Guide
title: Distribution and releases
description: Install RAGE in another project and publish releases.
tags:
  - installation
  - releases
---

# Distribution and releases

## External installation

External projects can install `rage` as a dependency.

In `requirements.txt`:

```text
rage>=<version>
```

Configure the RAGE release index in `uv.toml`:

```toml
[pip]
find-links = [
    "https://github.com/aureka-team/rage2/releases/expanded_assets/index",
]
```

## Releases

Pushing a Git tag matching `v*` runs
[`python-release.yml`](../.github/workflows/python-release.yml).

```bash
git tag v<version>
git push origin v<version>
```

The workflow builds the wheel, creates the GitHub release for the tag, and
uploads the wheel to both the tag release and the permanent `index` release used
by `uv`.

## License

RAGE is licensed under the terms of the [`LICENSE`](../LICENSE) file.
