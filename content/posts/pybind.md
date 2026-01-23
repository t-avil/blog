---
# Basic info
title: "INTRO: How to Build Your Own Version of NumPy"
date: 2026-01-21T23:27:20-08:00
draft: false
description: "From minimal pybind11 demos to vectorized CPU routines and CUDA acceleration learn to build Python C++ extensions and understand so files wheel distribution and Python ABI."
tags: ["cuda", "python", "numpy", "pybind","cpp", "SIMD","CUDA","OpenMP","scientific computing","extensions","packaging","abi"]
author: "Me" 
# Metadata & SEO
canonicalURL: "https://canonical.url/to/page"
hidemeta: false
searchHidden: true

# Table of contents
showToc: true
TocOpen: false
UseHugoToc: true

# Post features
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
ShowWordCount: true
comments: false

# Syntax highlighting
disableHLJS: false # set true to disable Highlight.js
disableShare: false

# Edit link
editPost:
  URL: "https://github.com/t-avil/blog/tree/main/content"
  Text: "Suggest Changes"   # edit link text
  appendFilePath: true      # append file path to the edit link
---

Python is expressive and flexible but **slow for heavy numeric computation**. Libraries like [NumPy](https://numpy.org/) combine Python with fast compiled backends to speed up arrays and matrix operations. With [Pybind11](https://pybind11.readthedocs.io/), you can write **C++ or CUDA code**, expose it to Python, and package it cross-platform.

In this post, we’ll cover building **vectorized CPU functions with SIMD and [OpenMP](https://www.openmp.org/)**, **GPU acceleration**, compiling `.so` modules, distributing via wheels, and understanding the [Python ABI](https://docs.python.org/3/c-api/abi.html). These examples serve as notes for anyone curious about how [CuPy](https://github.com/cupy/cupy), a GPU-powered NumPy alternative, works under the hood.


---

## 1 Environment setup

On Rocky Linux install required tools

```bash
sudo dnf install -y python3 python3-devel gcc gcc-c++ make cmake
pip3 install --user pybind11 wheel setuptools numpy
```

Check Python and compiler

```bash
python3 --version
gcc --version
python3 -m pybind11 --includes
```

---

## 2 Project structure

We will build two functions:
* `add_vectorized` optimized with SIMD, OpenMP, and BLAS
* `add_cuda` GPU accelerated

Directory layout:

```
$ tree -L 2
.
├── benchmark.py
├── example_cpu.cpp
├── example_cuda.cu
├── example (will be converted to example_src due to name collision)
    ├── example_cuda.so (we will compile this guy)
    └── __init__.py
├── Makefile
├── pyproject.toml
└── setup.py
```

---

## 3 Simple addition function

example.cpp

```cpp
#include <pybind11/pybind11.h>
namespace py = pybind11

int add_simple(int a int b) {
    return a + b
}

PYBIND11_MODULE(example m) {
    m.def("add_simple" &add_simple "Add two integers")
}
```

Build and test

```bash
python3 setup.py build_ext --inplace
python3
>>> import example
>>> example.add_simple(2 3)
5
```

This produces a `.so` shared object that Python dynamically loads.

---

## 4 Vectorized CPU function

High performance numeric code relies on **SIMD** (single instruction multiple data), **multi-core parallelism** (OpenMP), and **tuned linear algebra libraries** (BLAS/LAPACK).

### SIMD

Modern CPUs have vector registers AVX2, AVX-512, or NEON. Operations like addition, multiplication, or dot product can process multiple elements at once. You can use intrinsics or rely on compiler auto-vectorization. Example using AVX2:

```cpp
#include <immintrin.h>

void add_float_avx(float* a float* b float* out int n) {
    for (int i = 0 i < n i += 8) {
        __m256 va = _mm256_loadu_ps(a + i)
        __m256 vb = _mm256_loadu_ps(b + i)
        __m256 vc = _mm256_add_ps(va vb)
        _mm256_storeu_ps(out + i vc)
    }
}
```

### OpenMP

Parallelize loops across CPU cores:

```cpp
#pragma omp parallel for
for (int i = 0 i < N i++) {
    out[i] = a[i] + b[i]
}
```

### BLAS/LAPACK

For linear algebra rely on optimized libraries like OpenBLAS or MKL. Pybind11 can expose these routines to Python. Example: `cblas_sgemm` for matrix multiplication. We are not using these guys for now.

### Full Pybind example
./example_cpu.cpp

```cpp
#include <immintrin.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

py::array_t<float> add_vectorized(py::array_t<float> a, py::array_t<float> b) {
  auto buf_a = a.request();
  auto buf_b = b.request();

  if (buf_a.size != buf_b.size) {
    throw std::runtime_error("Arrays must be same size");
  }

  size_t n = buf_a.size;
  size_t n8 = (n / 8) * 8;

  auto result = py::array_t<float>(n);
  auto buf_r = result.request();

  float* ptr_a = static_cast<float*>(buf_a.ptr);
  float* ptr_b = static_cast<float*>(buf_b.ptr);
  float* ptr_r = static_cast<float*>(buf_r.ptr);

// AVX + OpenMP (canonical loop)
#pragma omp parallel for
  for (size_t i = 0; i < n8; i += 8) {
    __m256 va = _mm256_loadu_ps(ptr_a + i);
    __m256 vb = _mm256_loadu_ps(ptr_b + i);
    __m256 vr = _mm256_add_ps(va, vb);
    _mm256_storeu_ps(ptr_r + i, vr);
  }


  for (size_t i = n8; i < n; ++i) {
    ptr_r[i] = ptr_a[i] + ptr_b[i];
  }

  return result;
}

PYBIND11_MODULE(example_cpu, m) {
  m.def("add_vectorized", &add_vectorized,
        "Vectorized float addition with SIMD and OpenMP");
}
```

---

## 5 CUDA GPU function

./example_cuda.cu

```cpp
#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <stdexcept>

namespace py = pybind11;

__global__ void add_kernel(const float* a, const float* b, float* out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = a[idx] + b[idx];
  }
}

py::array_t<float> add_cuda(py::array_t<float> a, py::array_t<float> b) {
  auto buf_a = a.request();
  auto buf_b = b.request();

  if (buf_a.size != buf_b.size) {
    throw std::runtime_error("Arrays must have the same size");
  }

  int n = static_cast<int>(buf_a.size);

  float* d_a = nullptr;
  float* d_b = nullptr;
  float* d_out = nullptr;

  cudaMalloc(&d_a, n * sizeof(float));
  cudaMalloc(&d_b, n * sizeof(float));
  cudaMalloc(&d_out, n * sizeof(float));

  cudaMemcpy(d_a, buf_a.ptr, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, buf_b.ptr, n * sizeof(float), cudaMemcpyHostToDevice);

  int block = 256;
  int grid = (n + block - 1) / block;

  add_kernel<<<grid, block>>>(d_a, d_b, d_out, n);

  auto result = py::array_t<float>(n);
  cudaMemcpy(result.mutable_data(), d_out, n * sizeof(float),
             cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_out);

  return result;
}

PYBIND11_MODULE(example_cuda, m) {
  m.def("add_cuda", &add_cuda, "Add arrays on GPU using CUDA");
}

```

Compile with:

```bash
nvcc -O3 --compiler-options '-fPIC' -shared example_cuda.cu `python3 -m pybind11 --includes` -o example/example_cuda.so -lcudart
```

Python sees the `.so` as a normal module while execution happens on GPU

---

## 6 Understanding .so files, wheels, and Python ABI

Python ABI means **Application Binary Interface**. The chain is:

```
Python → CPython ABI → .so → C++ functions
```

Pybind11 leverages this chain. Using setuptools we can compile C++ code into a `.so` file that Python can load dynamically. A `.so` file is not a standalone program it is a **live binary module**:

* `.exe` standalone executable
* `.a` static library (copied in at compile time)
* `.so` shared library loaded at runtime

Alternatives:

* Linux `.so`
* macOS `.dylib`
* Windows `.dll`

### Anatomy of a `.so` file

A `.so` is an ELF file containing:

1. Machine code - CPU instructions not bytecode
2. Symbol table - exported functions and variables for linking
3. Relocation info - how addresses are fixed at load time
4. Dynamic section - dependencies on other `.so` files
5. Sections:

   * `.text` executable code
   * `.data` initialized globals
   * `.bss` uninitialized globals
   * `.rodata` constants

### Runtime usage

1. Program starts
2. Dynamic linker (`ld-linux.so`) loads `.so`
3. Symbols resolved
4. Code mapped into memory
5. Functions callable like normal code

Important note: `.so` files **cannot be moved across CPU architectures**

### Wheels

Wheels (`.whl`) are **Python distribution packages** that can include `.so` modules. They allow precompiled code to be installed with pip, avoiding the need for users to have a compiler:

```bash
python3 -m build
pip install dist/example-0.1.0-*.whl --force-reinstall
```

Key difference: `.so` is the **runtime artifact**. Wheel is the **distribution format** containing `.so` files, metadata, and Python package structure.

---

## 7 Setup script

setup.py

```python
from setuptools import setup, Extension
import pybind11
import os

cpu_ext = Extension(
    name="example.example_cpu",
    sources=["example_cpu.cpp"],
    include_dirs=[pybind11.get_include()],
    extra_compile_args=["-O3", "-march=native", "-fopenmp"],
    extra_link_args=["-fopenmp"],
    language="c++",
)




cuda_so = os.path.join("example", "example_cuda.so")
if not os.path.exists(cuda_so):
    raise RuntimeError(
        "example_cuda.so not found.\n"
        "Build it first with nvcc before running setup.py."
    )

setup(
    name="example",
    version="0.1.0",
    packages=["example"],
     package_data={
        "example": ["example_cuda.so"],
    },
    ext_modules=[cpu_ext],
    zip_safe=False,
)

```

---

## 8 Demo and Benchmark

```python
import example
import numpy as np
import time


# -----------------------------
# Config
# -----------------------------
N = 1_000_000_000
REPS = 10
PRINT_N = 10

# -----------------------------
# Prepare arrays
# -----------------------------
a = np.random.rand(N).astype(np.float32)
b = np.random.rand(N).astype(np.float32)

# -----------------------------
# CPU benchmark
# -----------------------------
cpu_times = []
for _ in range(REPS):
    start = time.time()
    c_cpu = example.add_vectorized(a, b)
    end = time.time()
    cpu_times.append(end - start)

avg_cpu = np.mean(cpu_times)
print(f"CPU first {PRINT_N} elements: {c_cpu[:PRINT_N]}")
print(f"CPU total time: {avg_cpu:.6f}s, per element: {avg_cpu/N*1e9:.3f} ns/element")

# -----------------------------
# GPU benchmark (total)
# -----------------------------
# Warm-up
example.add_cuda(a, b)

gpu_times = []
for _ in range(REPS):
    start = time.time()
    c_gpu = example.add_cuda(a, b)
    end = time.time()
    gpu_times.append(end - start)

avg_gpu_total = np.mean(gpu_times)
print(f"GPU first {PRINT_N} elements: {c_gpu[:PRINT_N]}")
print(f"GPU total time: {avg_gpu_total:.6f}s, per element: {avg_gpu_total/N*1e9:.3f} ns/element")

# -----------------------------
# Verify correctness
# -----------------------------
if np.allclose(c_cpu, c_gpu):
    print("CPU and GPU results match!")
else:
    print("CPU and GPU results do not match!")
```

*GPU overhead is significant for small arrays because of **memory transfer latency**. With larger arrays, GPU naturally outperforms due to massive parallelization.*


{{< figure src="/images/pybind_cuda_cpu_stupid_benchmark.jpg" attr="Benchmark results at 1B floats for cpu/gpu simple examples in pybind." target="_blank" >}}

---

## 9 Makefile & Pyproject.toml

**Makefile**:

```make
all: run

example/example_cuda.so: example_cuda.cu
	mkdir -p example
	nvcc -O3 --compiler-options '-fPIC' -shared example_cuda.cu `python3 -m pybind11 --includes` -o example/example_cuda.so -lcudart

example/__init__.py: example/example_cuda.so
	echo "from .example_cuda import *" > example/__init__.py
	echo "from .example_cpu import *" >> example/__init__.py

wheel: example/__init__.py
	python3 -m build
	mv ./example ./example_src

install: wheel
	pip install dist/example-0.1.0-*.whl --force-reinstall

run: install
	python3 ./benchmark.py

clean:
	pip uninstall example -y
	rm -rf build dist example.egg-info example-0.1.0 example
```

**pyproject.toml**:

```toml
[build-system]
requires = [
    "setuptools>=64",
    "wheel",
    "pybind11>=2.10"
]
build-backend = "setuptools.build_meta"
```

This allows building the package with `python3 build` and distributing as a wheel.

---

## 10 Python Package Lifecycle & Dependency Management

Understanding how Python modules, packages, compiled extensions, and dependencies interact is crucial when building libraries like your own NumPy clone. Here’s a detailed breakdown:

### 10.1 Modules and Packages

* **Module**: a single `.py` file containing Python code-functions, classes, constants-that can be imported. Example:

```python
# utils.py
def add(a, b):
    return a + b
```

* **Package**: a directory with an `__init__.py` file (even if empty) that tells Python “this is a package.” Packages can contain multiple modules and sub-packages, creating a hierarchical structure:

```
mypackage/
├── __init__.py
├── utils.py
└── subpackage/
    ├── __init__.py
    └── math_ops.py
```

* Packages may also include:

  * **Compiled extensions** (`.so` on Linux, `.pyd` on Windows), typically generated via **pybind11** or **Cython**.
  * **Data files** (images, JSON, CSV, etc.)
  * **Metadata files** (`setup.py`, `pyproject.toml`, `MANIFEST.in`) describing the package, version, dependencies, and entry points.

---

### 10.2 Installation Process

When you install a package locally or via PyPI:

```bash
pip install ./mypackage
```

1. **Metadata parsing**: Pip reads `setup.py` or `pyproject.toml` to discover package info and dependencies.
2. **Dependency resolution**: Pip determines which other packages are needed, including versions.
3. **Compilation of extensions**: Any `.cpp` or `.cu` files are compiled into `.so`/`.pyd` using the system compiler.
4. **Copying files**: Python code, compiled binaries, and package data are copied into the `site-packages` directory of your environment.
5. **Isolation via virtual environments**: If you’re inside a `venv`, this installation is isolated from the system Python. This allows multiple versions of the same package in different projects without conflict.
6. **Metadata storage**: Pip generates `.dist-info` folders containing package metadata, dependencies, and file records for uninstallation or introspection.

---

### 10.3 Wheels and Distribution

* **Wheel (`.whl`)**: a standardized binary distribution format for Python. Wheels can include precompiled extensions so users **don’t need a compiler**.
* Platform-specific if compiled extensions exist; pure Python wheels work across all platforms.
* **Source distribution (`.tar.gz`)** contains the raw Python/C++ source and requires compilation at install time.
* Uploading:

```bash
twine upload dist/mypackage-0.1.0-py3-none-any.whl
```

* Users can then install via:

```bash
pip install mypackage
```

* The wheel contains everything pip needs: Python code, compiled binaries, metadata, and optional console scripts.

---

### 10.4 Python ABI and Compiled Extensions

* **Python ABI (Application Binary Interface)** defines how compiled extensions (.so or .pyd) interact with the Python interpreter.

* Pybind11 wraps C++ or CUDA code and exposes it through the CPython ABI so that Python can **dynamically load functions at runtime**.

* Life cycle for compiled extensions:

  1. `.so` or `.pyd` is compiled.
  2. Python imports the module.
  3. The system loader dynamically maps the machine code into memory.
  4. Python calls functions in the extension as if they were regular Python functions.

* `.so` files **cannot be moved across CPU architectures**; they must be recompiled for each target CPU.

---

### 10.5 Dependency Management

Modern Python projects often include external dependencies. Dependencies can be:

* **PyPI packages** (e.g., numpy, scipy)
* **Git repositories** (private or public)

Example in `pyproject.toml` or `requirements.txt`:

```
git+https://github.com/username/mylib.git@v1.2.3#egg=mylib
```

* When pip installs this dependency:

  * It fetches the repository.
  * Checks out the specified commit, branch, or tag.
  * Installs it alongside standard PyPI packages.
* Pip itself **does not resolve Git versions from lock files** (like `uv.lock`); it only installs the URL you provide.
* Tools like **Poetry** or **pip-tools** sit on top of pip:

  * They generate **lock files** (`poetry.lock` or `requirements.txt`) that pin exact versions and commits.
  * They ensure reproducible installations across environments.
* Version specifiers:

  * `==1.2.3` → exact version
  * `>=1.2.0,<2.0.0` → compatible range
  * `~=1.2.3` → compatible with 1.2.x patches

Unlike Node.js/npm, Python does **not deduplicate versions automatically**; multiple versions may coexist in `site-packages`. Tools like Poetry or uv resolve versions into a deterministic environment.

---

### 10.6 Full Lifecycle

1. **Develop** your module/package.
2. **Structure** it with `__init__.py`, metadata, optional compiled extensions, and data files.
3. **Test locally** in a virtual environment.
4. **Build** sdist or wheel.
5. **Publish** to PyPI or private repository.
6. **Install** in another environment with pip.
7. Python handles **module resolution, dynamic loading**, and **dependency management**.

*Extra notes*:

* Packages can include **console scripts** (entry points), type stubs, and runtime-accessible data.
* Compiled extensions bridge Python with high-performance C++/CUDA code.
* With proper packaging, users can run your library without a compiler, achieving both **ease of installation** and **high performance**.

---