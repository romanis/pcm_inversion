"""Compiled kernel of the harness: ``pcm_kernel`` (python_bindings/pcm_kernel_python.cpp, DESIGN 1.2, 3).

The shared object is a build output and is not tracked. Build it with:

    cmake -S /local/home/istomin/pcm_inversion -B /local/home/istomin/pcm_inversion/build-rel2 \\
          -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE=/home/istomin/anaconda3/bin/python \\
          -Dpybind11_DIR=/home/istomin/.local/lib/python3.12/site-packages/pybind11/share/cmake/pybind11
    cmake --build /local/home/istomin/pcm_inversion/build-rel2 -j 64 --target pcm_kernel_py
"""

try:
    from . import pcm_kernel
except ImportError as exc:  # pragma: no cover - depends on the build
    raise ImportError(
        "nfp_harness._native.pcm_kernel is not built. Build the CMake target pcm_kernel_py "
        "(see the docstring of nfp_harness/_native/__init__.py or docs/estimation_harness/DESIGN.md 1.2); "
        f"it writes pcm_kernel.cpython-*.so into nfp_harness/_native. Original error: {exc}"
    ) from exc

__all__ = ["pcm_kernel"]
