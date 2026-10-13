#!/usr/bin/env bash
# Check whether a versioned image exists, or wait for its build to publish it.
# Usage: bash ecr-image.sh REPOSITORY TAG [exists|wait]
# Requires ECR_REGISTRY and AWS credentials with ecr:DescribeImages permission.
# Emits exists and, when found, digest to GITHUB_OUTPUT or stdout locally.
# In exists mode, a missing image returns exists=false successfully. In wait
# mode, an unavailable image or any other AWS error exits with status 1.
set -euo pipefail

repository=${1:?Pass an ECR repository}
tag=${2:?Pass an image tag}
mode=${3:-exists}
attempts=1
# Release publication can race the tag build: retry every 10 seconds, up to
# 30 lookups. Build workflows use a single lookup to avoid overwriting images.
[[ "$mode" == wait ]] && attempts=30
error_file=$(mktemp)
trap 'rm -f "$error_file"' EXIT
for ((attempt=1; attempt<=attempts; attempt++)); do
  # The hostname starts with the AWS account ID used by ECR's registry API.
  if digest=$(aws ecr describe-images \
    --registry-id "${ECR_REGISTRY%%.*}" \
    --repository-name "$repository" --image-ids "imageTag=$tag" \
    --query 'imageDetails[0].imageDigest' --output text 2>"$error_file"); then
    {
      echo 'exists=true'
      echo "digest=$digest"
    } >> "${GITHUB_OUTPUT:-/dev/stdout}"
    exit 0
  fi
  # Permission or network failures must not be treated as an unpublished image.
  if ! grep -q 'ImageNotFoundException' "$error_file"; then
    cat "$error_file" >&2
    exit 1
  fi
  if [[ "$mode" != wait ]]; then
    echo 'exists=false' >> "${GITHUB_OUTPUT:-/dev/stdout}"
    exit 0
  fi
  if ((attempt < attempts)); then
    echo "Waiting for $repository:$tag ($attempt/$attempts)" >&2
    sleep 10
  fi
done
echo "Image $repository:$tag was not published; deployment stopped" >&2
exit 1
