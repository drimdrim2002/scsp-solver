#!/usr/bin/env bash
# Build and test scsp in a linux/amd64 container (g++, x86-64).
# Needs a running docker daemon (on this Mac: `colima start`).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
docker build --platform linux/amd64 -f "$ROOT/docker/Dockerfile.linux-amd64" -t scsp-linux-amd64 "$ROOT"
docker run --rm --platform linux/amd64 scsp-linux-amd64
