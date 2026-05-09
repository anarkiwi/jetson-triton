# jetson-triton

Triton wheel + image for NVIDIA Jetson (linux/arm64, Orin sm_87),
split out from `anarkiwi/jetson-pytorch` so the heavy LLVM compile no
longer has to share a runner with the pytorch build.

The Triton compile peaks at ~16 GB during LLVM linking, which on the
GitHub-hosted `ubuntu-24.04-arm` runner (ARM-native, 16 GB) ran the
combined pytorch+triton build out of memory mid-step. Triton alone
fits comfortably on the same runner; pytorch alone fits too. Same
ABI guarantee is preserved by reading the triton commit pin directly
from torch's CI metadata.

## Output

* **Image**: `anarkiwi/jetson-triton:${VERSION}` -- minimal
  l4t-jetpack base with triton pip-installed and the source wheel at
  `/triton/dist/triton-*.whl`.
* **Wheel** (inside the image): consumers can extract via
  multi-stage:

  ```dockerfile
  COPY --from=anarkiwi/jetson-triton:vX.Y.Z /triton/dist /triton/dist
  RUN pip install /triton/dist/*.whl
  ```

  Or bind-mount during `RUN`:

  ```dockerfile
  RUN --mount=type=bind,from=anarkiwi/jetson-triton:vX.Y.Z,source=/triton/dist,target=/triton/dist \
      pip install /triton/dist/*.whl
  ```

## Versioning

The image tag matches the corresponding `anarkiwi/jetson-pytorch`
version (e.g. `v2.11.0`). The triton commit is derived from
`pytorch/.ci/docker/ci_commit_pins/triton.txt` at that pytorch tag,
so the wheel ABI tracks torch's expectations automatically.

## Local build (defroster, native arm64 via QEMU)

The pip mirror env var matches the rest of the anarkiwi setup:

```bash
export PIP_OPTS="--index-url http://192.168.5.1:5001/index/ --trusted-host 192.168.5.1"
./build.sh                  # default v2.11.0
./build.sh v2.10.0          # override
```

`build.sh` invokes `docker buildx --platform linux/arm64`. On
defroster (94 GB RAM, x86_64) the linux/arm64 platform runs through
QEMU; the LLVM compile fits well within budget. On a native arm64
host (Orin NX, Ampere VM, GH `ubuntu-24.04-arm` runner) the same
command compiles natively.

## Release flow

`.github/workflows/release.yml` mirrors the jetson-pytorch one:

1. Push a tag (`vX.Y.Z`) to trigger the workflow.
2. Workflow builds + pushes to Docker Hub as
   `anarkiwi/jetson-triton:vX.Y.Z`.
3. Branch `main` publishes as `anarkiwi/jetson-triton:latest`.

Required secrets in the repo's `release` environment:

* `DOCKER_USERNAME`
* `DOCKER_PASSWORD`

## Build DAG

```
l4t-jetpack → jetson-pytorch → jetson-triton → consumer (preframr-jetson, etc.)
              (torch wheel,      (FROM ↑;
              no triton)         builds triton
                                 against installed
                                 torch)
```

One-way. **jetson-pytorch must be built and published first**, with
its inline triton-builder stage removed (this repo replaces it).
jetson-triton then `FROM`s the no-triton jetson-pytorch image and
produces the triton wheel.

Closing this into a cycle (e.g. having jetson-pytorch install
triton from anarkiwi/jetson-triton in its own final stage) would
deadlock the first-publish bootstrap; don't do that. Instead, leave
jetson-pytorch as torch-only and have downstream consumer images
pull both pieces:

```dockerfile
# Option A: FROM jetson-triton (carries torch + triton already).
FROM anarkiwi/jetson-triton:vX.Y.Z

# Option B: FROM jetson-pytorch, install triton wheel via
# multi-stage from jetson-triton.
FROM anarkiwi/jetson-pytorch:vX.Y.Z
RUN --mount=type=bind,from=anarkiwi/jetson-triton:vX.Y.Z,source=/triton/dist,target=/triton/dist \
    pip install --no-cache-dir /triton/dist/*.whl
```

Each repo's build fits its own 16 GB ARM-native runner; the failed
combined v2.11.0 pattern is gone.

### Bootstrap order (first time at a new pytorch version)

1. In `anarkiwi/jetson-pytorch`'s `Dockerfile.pytorch`, drop the
   `triton-builder` stage and the triton wheel install from the
   final stage. Bump the tag, push -- builds + publishes
   `anarkiwi/jetson-pytorch:vX.Y.Z` (torch-only).
2. Tag this repo at `vX.Y.Z`, push -- workflow builds against (1)
   and publishes `anarkiwi/jetson-triton:vX.Y.Z`.
3. Update consumer Dockerfiles (e.g. `anarkiwi/preframr` jetson
   branch's `Dockerfile`) to reference both as above. The
   jetson-pytorch image now has no triton; consumers that need it
   pull it from here.

## Sanity check

```bash
docker run --rm anarkiwi/jetson-triton:vX.Y.Z \
    python3 -c "import triton; print(triton.__version__)"
```
