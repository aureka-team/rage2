#!/usr/bin/env bash
# Check tag parsing and ECR lookup behavior without AWS access or deployments.
# Usage from the repository root: bash .github/scripts/test-release-scripts.sh
# The image workflow runs this before publishing images. A failed assertion
# exits nonzero; success prints "Release script checks passed".
set -euo pipefail
scripts=$(cd "$(dirname "$0")" && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
export GITHUB_OUTPUT="$tmp/output"

# Each row specifies the accepted tag, required channel, and Python version.
while read -r tag channel version; do
  : > "$GITHUB_OUTPUT"
  bash "$scripts/tag-version.sh" "$tag" "$channel"
  grep -qx "tag=$tag" "$GITHUB_OUTPUT"
  grep -qx "channel=$channel" "$GITHUB_OUTPUT"
  grep -qx "package_version=$version" "$GITHUB_OUTPUT"
done <<'CASES'
v1.2.3 stable 1.2.3
v0.0.0 stable 0.0.0
v1.2.3-rc.1 rc 1.2.3rc1
v1.2.3-eks.12 eks 1.2.3.dev12
v1.2.3-eks.0 eks 1.2.3.dev0
CASES
# Reject malformed versions, unsupported suffixes, and shell-like tag text.
for tag in main v1.2 v1.2.3.4 v01.2.3 v1.2.3-rc1 v1.2.3-eks.01 \
  v1.2.3-rc.1-extra v1.2.3-beta.1 'v1.2.3;echo bad'; do
  if bash "$scripts/tag-version.sh" "$tag" 2>/dev/null; then
    echo "Invalid tag accepted: $tag" >&2
    exit 1
  fi
done
# Production and staging must reject tags belonging to another channel.
for tag in v1.2.3-rc.1 v1.2.3-eks.1; do
  if bash "$scripts/tag-version.sh" "$tag" stable 2>/dev/null; then
    echo "Candidate accepted as stable: $tag" >&2
    exit 1
  fi
done
if bash "$scripts/tag-version.sh" v1.2.3 rc 2>/dev/null; then
  echo 'Stable tag accepted as RC' >&2
  exit 1
fi

# Exercise ECR lookup/retry behavior without AWS credentials or waiting.
mkdir "$tmp/bin"
cat > "$tmp/bin/aws" <<'MOCK'
#!/usr/bin/env bash
set -euo pipefail
case "$MOCK_ECR" in
  found) echo sha256:example ;;
  denied) echo AccessDeniedException >&2; exit 1 ;;
  delayed)
    if [[ -f "$MOCK_STATE" ]]; then echo sha256:example; exit 0; fi
    touch "$MOCK_STATE"
    echo ImageNotFoundException >&2; exit 1 ;;
  missing) echo ImageNotFoundException >&2; exit 1 ;;
esac
MOCK
cat > "$tmp/bin/sleep" <<'MOCK'
#!/usr/bin/env bash
exit 0
MOCK
chmod +x "$tmp/bin/aws" "$tmp/bin/sleep"
# Override commands only in this script's process. The AWS mock returns the
# selected scenario; the sleep mock makes retries complete immediately.
export PATH="$tmp/bin:$PATH"
export ECR_REGISTRY=123456789012.dkr.ecr.eu-central-1.amazonaws.com
export MOCK_STATE="$tmp/state"
# An existing image supplies its digest; a missing image can be built.
export MOCK_ECR=found
: > "$GITHUB_OUTPUT"
bash "$scripts/ecr-image.sh" example v1.2.3
grep -qx 'exists=true' "$GITHUB_OUTPUT"
grep -qx 'digest=sha256:example' "$GITHUB_OUTPUT"
export MOCK_ECR=missing
: > "$GITHUB_OUTPUT"
bash "$scripts/ecr-image.sh" example v1.2.3
grep -qx 'exists=false' "$GITHUB_OUTPUT"
# Release deployment can proceed when an image appears on a later lookup.
export MOCK_ECR=delayed
: > "$GITHUB_OUTPUT"
bash "$scripts/ecr-image.sh" example v1.2.3 wait 2>/dev/null
grep -qx 'exists=true' "$GITHUB_OUTPUT"
# An image that never appears, or an AWS access error, must stop deployment.
for MOCK_ECR in missing denied; do
  export MOCK_ECR
  if bash "$scripts/ecr-image.sh" example v1.2.3 wait 2>/dev/null; then
    echo "ECR $MOCK_ECR allowed deployment" >&2
    exit 1
  fi
done
echo 'Release script checks passed'
