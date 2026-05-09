#!/bin/bash
# Local arm64 build of anarkiwi/jetson-triton on defroster (or any
# x86_64 host with docker buildx + QEMU registered for linux/arm64).
#
# Usage:
#   export PIP_OPTS="--index-url http://192.168.5.1:5001/index/ --trusted-host 192.168.5.1"
#   ./build.sh              # default PYTORCH_VERSION=v2.11.0
#   ./build.sh v2.10.0      # override
#
# Output image: anarkiwi/jetson-triton:${PYTORCH_VERSION}.
set -e

PYTORCH_VERSION=${1:-v2.11.0}

docker buildx build \
    --platform linux/arm64 \
    --build-arg PIP_OPTS="${PIP_OPTS}" \
    --build-arg PYTORCH_VERSION="${PYTORCH_VERSION}" \
    --tag "anarkiwi/jetson-triton:${PYTORCH_VERSION}" \
    --load \
    -f Dockerfile \
    .
