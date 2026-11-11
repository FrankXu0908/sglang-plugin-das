#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

# Build the hcu-sglang-plugin wheel from the single version source
# (pyproject.toml [project].version). One command, one artifact:
#   dist/hcu_sglang_plugin-<version>+hcu.<stamp>-*.whl
# --no-stamp builds the raw version line without the time/commit local label.
set -euo pipefail

USE_STAMP=1
for arg in "$@"; do
    case "$arg" in
        --no-stamp) USE_STAMP=0 ;;
        -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
        *) echo "build.sh: unknown argument: $arg" >&2; exit 2 ;;
    esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="$(sed -n 's/^version = "\(.*\)"$/\1/p' "$ROOT/pyproject.toml" | head -1)"
if [ -z "$VERSION" ]; then
    echo "build.sh: cannot read version from pyproject.toml" >&2
    exit 1
fi

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
    # probe: some Windows boxes ship a Microsoft-Store python3 stub that
    # exists on PATH but cannot run
    for cand in python3 python; do
        if "$cand" -c "import sys" >/dev/null 2>&1; then PY="$cand"; break; fi
    done
fi
[ -n "$PY" ] || { echo "build.sh: no working python3/python on PATH" >&2; exit 1; }

# pip wheel --no-build-isolation needs the build backend in this environment
if ! "$PY" -c "import setuptools.build_meta" >/dev/null 2>&1; then
    echo "== installing build deps (setuptools/wheel) =="
    "$PY" -m pip install "setuptools>=64" wheel
fi

mkdir -p "$ROOT/dist"
BUILD="$ROOT"
if [ "$USE_STAMP" = 1 ]; then
    # one stamp per build: <YYMMDDHHMM>.g<commit6> (rendered as hcu.<stamp>),
    # no dtk/torch parts; pyproject.toml carries a static version, so the
    # stamped build uses a throwaway copy and never edits the tracked file
    DAS_BUILD_STAMP="${DAS_BUILD_STAMP:-$(date +%y%m%d%H%M).g$(git -C "$ROOT" rev-parse --short=6 HEAD)}"
    BUILD_VERSION="${VERSION%%+*}+hcu.${DAS_BUILD_STAMP}"
    BUILD="$(mktemp -d)"
    trap 'rm -rf "$BUILD"' EXIT
    tar -C "$ROOT" \
        --exclude=.git --exclude=build --exclude=dist \
        --exclude='*.egg-info' --exclude=__pycache__ --exclude='*.pyc' \
        -cf - . | tar -C "$BUILD" -xf -
    sed -i "s/^version = \".*\"\$/version = \"${BUILD_VERSION}\"/" "$BUILD/pyproject.toml"
    if ! grep -qxF "version = \"${BUILD_VERSION}\"" "$BUILD/pyproject.toml"; then
        echo "build.sh: failed to stamp version in the build copy" >&2
        exit 1
    fi
else
    BUILD_VERSION="$VERSION"
fi

echo "== building hcu-sglang-plugin ${BUILD_VERSION} ($BUILD) =="
"$PY" -m pip wheel --no-deps --no-build-isolation \
    --wheel-dir "$ROOT/dist" "$BUILD"

WHEEL="$(ls -t "$ROOT/dist"/hcu_sglang_plugin-"${BUILD_VERSION}"-*.whl | head -1)"
sha256sum "$WHEEL"
