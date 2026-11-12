#!/usr/bin/env bash
# Validate a deployment tag and emit its release channel and Python version.
# Usage: bash tag-version.sh TAG [any|stable|rc|eks]
# Writes tag, channel, and package_version as key=value lines to GITHUB_OUTPUT,
# or stdout locally. Invalid tags or channel mismatches exit with status 1.
set -euo pipefail

tag=${1:?Pass a release tag}
expected=${2:-any}
# SemVer numeric segments cannot have leading zeros, including candidate N.
number='(0|[1-9][0-9]*)'
if [[ ! "$tag" =~ ^v${number}\.${number}\.${number}(-(rc|eks)\.${number})?$ ]]; then
  echo "Expected vX.Y.Z, vX.Y.Z-rc.N, or vX.Y.Z-eks.N" >&2
  exit 1
fi
# Read the regex captures before another regex could replace BASH_REMATCH.
version="${BASH_REMATCH[1]}.${BASH_REMATCH[2]}.${BASH_REMATCH[3]}"
channel=${BASH_REMATCH[5]:-stable}
sequence=${BASH_REMATCH[6]:-}
if [[ "$expected" != any && "$channel" != "$expected" ]]; then
  echo "Expected a $expected tag; got $channel" >&2
  exit 1
fi
# Keep the image tag unchanged; convert its suffix for Python packaging.
# setuptools-scm needs a PEP 440 version for tests and container builds.
case "$channel" in
  rc) version+="rc$sequence" ;;
  eks) version+=".dev$sequence" ;;
esac
{
  echo "tag=$tag"
  echo "channel=$channel"
  echo "package_version=$version"
} >> "${GITHUB_OUTPUT:-/dev/stdout}"
