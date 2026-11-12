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

| Trigger | Outcome |
| --- | --- |
| Push `vX.Y.Z-rc.N` | Deploy the RC image to Floo staging |
| Push `vX.Y.Z` | Deploy the stable image to EKS through Flux, starting at `v3.0.5` |
| Publish a GitHub Release for `vX.Y.Z` | Publish the Python wheel |

[`image.yml`](../.github/workflows/image.yml) validates the tag, checks
source compilation and package construction, builds the `api` target from the
shared [`Dockerfile`](../Dockerfile), and pushes it to
`696036763958.dkr.ecr.eu-central-1.amazonaws.com/rage-api` using the exact Git
tag. Reruns reuse an existing image. RC tags map to `X.Y.ZrcN` inside the Python
package. The original tag remains the container
tag. Pushes to `main` run package checks and do not publish container images.

Push a `vX.Y.Z-rc.N` tag to deploy to Floo staging. After the image build
succeeds, [the staging workflow](../.github/workflows/deploy-staging.yml)
reads the tag from that build and rolls out the RAGE stack on EC2. Before
the first RC, configure the staging SSH settings and provision the RAGE stack
files, runtime credentials, and private network on the Floo host.

To deploy to EKS, push a stable tag and wait for the image workflow to
succeed. Flux selects the highest eligible stable image in ECR.

To release the Python package, publish a GitHub Release for that existing tag
with pre-release disabled. [`python-release.yml`](../.github/workflows/python-release.yml) builds
the wheel and uploads it to the published release and the permanent `index`
release used by `uv`. It accepts only exact `vX.Y.Z` tags and does not create the
version release itself.

Confirm that the image and package workflows succeed. For the complete app
deployment and EKS cutover procedure, see the
[aureka deployment guide](https://github.com/aureka-team/gitops-aureka/blob/main/docs/deployments.md).

## License

RAGE is licensed under the terms of the [`LICENSE`](../LICENSE) file.
