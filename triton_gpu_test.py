"""On-device check that triton compiles and runs kernels on a real Orin.

Counterpart to triton_smoke_test.py, which deliberately avoids the GPU so
CI can gate packaging. Run under the nvidia runtime; the JIT invokes
ptxas for sm_87 here, which nothing off-hardware exercises.
"""

import sys

import torch
import triton
import triton.language as tl

BLOCK = 1024
N = 98432


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, block: tl.constexpr):
    """Elementwise add, the minimal kernel that proves codegen works."""
    off = tl.program_id(0) * block + tl.arange(0, block)
    mask = off < n
    x = tl.load(x_ptr + off, mask=mask)
    y = tl.load(y_ptr + off, mask=mask)
    tl.store(out_ptr + off, x + y, mask=mask)


def main() -> int:
    """Compile a kernel via triton, then via torch.compile's inductor path."""
    assert torch.cuda.is_available(), "no CUDA device visible"
    print(f"triton  {triton.__version__}")
    print(f"device  {torch.cuda.get_device_name(0)}")

    x = torch.rand(N, device="cuda")
    y = torch.rand(N, device="cuda")
    out = torch.empty_like(x)
    add_kernel[(triton.cdiv(N, BLOCK),)](x, y, out, N, block=BLOCK)
    assert torch.allclose(out, x + y), "triton kernel result mismatch"
    print("kernel  OK")

    compiled = torch.compile(lambda a, b: (a * b).relu().sum())
    a = torch.randn(512, 512, device="cuda")
    b = torch.randn(512, 512, device="cuda")
    assert torch.allclose(compiled(a, b), (a * b).relu().sum(), atol=1e-3)
    print("compile OK")

    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
