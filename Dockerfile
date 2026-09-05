# Build a Triton wheel for Jetson (linux/arm64, sm_87 Orin) split out
# from anarkiwi/jetson-pytorch's monolithic build. The combined
# pytorch + triton compile peaks above the 16 GB ARM-native runner's
# budget (LLVM linking is the offender); building triton standalone
# here lets each image fit its own runner, and lets the Triton wheel
# be consumed by other downstream images via multi-stage COPY.
#
# Output:
#   /triton/dist/triton-*.whl  -- copied into the final stage so
#                                  consumers can mount it via:
#       COPY --from=anarkiwi/jetson-triton:vX.Y.Z /triton/dist /triton/dist
#       RUN pip install /triton/dist/*.whl
#
# Triton commit selection follows the same rule as pytorch's own CI:
# the file ``.ci/docker/ci_commit_pins/triton.txt`` in the pytorch
# source tree at the matching version names the exact commit. We
# shallow-clone pytorch just to read that pin, so the produced wheel
# is ABI-matched to the torch version the consumer image carries.

ARG PYTORCH_VERSION="v2.14.0"

# Stage 1: derive the triton commit pinned by torch's CI for the target
# pytorch version. Sparse + filter=blob:none keeps this clone small
# (just the metadata file we need). Plain ubuntu here -- this stage
# needs git and nothing else, and matches the 24.04 userspace of the
# JetPack 7.2 / CUDA 13.2 SBSA base that jetson-pytorch now builds on.
FROM ubuntu:24.04 AS pin-resolver
ARG PYTORCH_VERSION
RUN apt-get -yq update && apt-get install --no-install-recommends -yq git ca-certificates
RUN git clone --depth 1 --branch ${PYTORCH_VERSION} \
        --filter=blob:none --sparse \
        https://github.com/pytorch/pytorch /pytorch \
    && cd /pytorch \
    && git sparse-checkout set .ci/docker/ci_commit_pins \
    && cp .ci/docker/ci_commit_pins/triton.txt /triton_pin.txt

# Stage 2: build the triton wheel against the matching torch version.
# anarkiwi/jetson-pytorch's published image has torch installed, which
# triton's setup.py imports to derive ABI / link paths. The triton
# already in the base image (if any) is irrelevant -- ``pip wheel .``
# builds from the source tree.
FROM anarkiwi/jetson-pytorch:${PYTORCH_VERSION} AS triton-builder
ARG PIP_OPTS=""
ENV PIP_OPTS=$PIP_OPTS
ENV PIP_BREAK_SYSTEM_PACKAGES=1
RUN apt-get -yq update && apt-get install --no-install-recommends -yq \
        git python3-pip libssl-dev cmake ninja-build \
        g++ zlib1g-dev libstdc++-13-dev \
    && pip install $PIP_OPTS --no-cache-dir wheel scikit-build ninja lit
COPY --from=pin-resolver /triton_pin.txt /tmp/triton_pin.txt
RUN git clone https://github.com/triton-lang/triton.git /triton \
    && cd /triton \
    && git checkout "$(cat /tmp/triton_pin.txt)" \
    && git submodule update --init --recursive
WORKDIR /triton
# MAX_JOBS=2 keeps the LLVM build under ~12 GB peak; bump on a beefier
# runner. TORCH_CUDA_ARCH_LIST="8.7" targets Orin (sm_87); add 7.2 if
# Xavier support is wanted.
RUN MAX_JOBS=2 TORCH_CUDA_ARCH_LIST="8.7" \
        pip wheel . -w /triton/dist --no-deps --no-cache-dir $PIP_OPTS

# Stage 3: release image. Inherits torch from anarkiwi/jetson-pytorch
# so ``import triton`` (which links against torch_python at load-time)
# and ``import torch`` both work out of the box. Carries the
# freshly-built wheel at /triton/dist/ so consumers can extract it
# via ``COPY --from``, and pip-installs the same wheel over whatever
# triton the base may already carry. Carries its own tests alongside
# the /smoke_test.py and /gpu_test.py inherited from jetson-pytorch,
# which still cover the torch half.
FROM anarkiwi/jetson-pytorch:${PYTORCH_VERSION}
ARG PIP_OPTS=""
ENV PIP_OPTS=$PIP_OPTS
ENV PIP_BREAK_SYSTEM_PACKAGES=1
COPY --from=triton-builder /triton/dist /triton/dist
RUN pip install $PIP_OPTS --no-cache-dir --force-reinstall --no-deps /triton/dist/*.whl
COPY triton_smoke_test.py triton_gpu_test.py /
