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

Push a tag using one of the shared release formats:

| Tag | Container image | Python package |
| --- | --- | --- |
| `vX.Y.Z-eks.N` | EKS candidate selected by Flux | No release |
| `vX.Y.Z-rc.N` | RC image, without an EC2 deployment | No release |
| `vX.Y.Z` | Stable image, without automatic deployment | Waits for GitHub Release publication |

To publish an EKS candidate, create and push a tag:

```bash
git tag v3.0.2-eks.1
git push origin v3.0.2-eks.1
```

Replace the example version with the next release version. Increment `N` for
each candidate and use numeric segments without leading zeros.

[`image.yml`](../.github/workflows/image.yml) validates the tag, checks
source compilation and package construction, builds the `api` target from the
shared [`Dockerfile`](../Dockerfile), and pushes it to
`696036763958.dkr.ecr.eu-central-1.amazonaws.com/rage-api` using the exact Git
tag. Reruns reuse an existing image. EKS tags map to `X.Y.Z.devN` inside the
Python package; RC tags map to `X.Y.ZrcN`. The original tag remains the container
tag. Pushes to `main` run package checks and do not publish container images.

To release the Python package, push a stable tag, wait for the image workflow to
succeed, and publish a GitHub Release for that existing tag with pre-release
disabled. [`python-release.yml`](../.github/workflows/python-release.yml) builds
the wheel and uploads it to the published release and the permanent `index`
release used by `uv`. It accepts only exact `vX.Y.Z` tags and does not create the
version release itself.

Confirm that the image and package workflows succeed. For the complete app
deployment and EKS cutover procedure, see the
[aureka deployment guide](https://github.com/aureka-team/gitops-aureka/blob/main/docs/deployments.md).

## License

RAGE is licensed under the terms of the [`LICENSE`](../LICENSE) file.
