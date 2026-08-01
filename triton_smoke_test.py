"""Post-build check that the triton wheel is installed and linked.

Runs without a driver so GPU-less CI gates packaging. Compiling a kernel
needs real hardware -- see triton_gpu_test.py. The torch half of the
image is covered by /smoke_test.py, inherited from jetson-pytorch.
"""

import glob
import shutil
import sys

import torch
import triton
from triton.backends import backends

EXPECTED_TRITON_MAJOR = 3


def main() -> int:
    """Report versions and assert the built wheel is what got installed."""
    print(f"torch    {torch.__version__}")
    print(f"triton   {triton.__version__}")
    print(f"backends {sorted(backends)}")

    assert triton.__version__.split(".")[0] == str(
        EXPECTED_TRITON_MAJOR
    ), triton.__version__
    assert "nvidia" in backends, sorted(backends)

    wheels = glob.glob("/triton/dist/triton-*.whl")
    assert wheels, "no triton wheel at /triton/dist for consumers to COPY"
    wheel = wheels[0].rsplit("/", 1)[-1]
    print(f"wheel    {wheel}")
    assert wheel.startswith(f"triton-{triton.__version__}"), (wheel, triton.__version__)

    # JIT shells out to ptxas, hence the CUDA devel base rather than runtime.
    ptxas = shutil.which("ptxas")
    print(f"ptxas    {ptxas}")
    assert ptxas, "no ptxas on PATH; triton cannot compile kernels"

    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
