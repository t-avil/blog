---
# Basic info
title: "FAST_GUIDE: How to Build Your Own Version of NumPy"
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

Python is expressive and flexible but slow for heavy numeric computation. Libraries like NumPy exist because someone realized Python is great for expressing ideas but too slow for matrix multiplications or numerical loops. Pybind11 allows us to write **C++ code, expose it to Python, and package it in a cross-platform way**.

In this post we will go from **environment setup**, to **writing three pybind11 functions** (simple add, vectorized CPU, GPU CUDA), to **deep understanding of .so files, Python packaging, wheels, and ABI compatibility**, showing how everything connects under the hood.

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

We will build three functions:

* `add_simple` minimal integer addition
* `add_vectorized` optimized with SIMD, OpenMP, and BLAS
* `add_cuda` GPU accelerated

Directory layout:

```
pybind_demo
setup.py
example.cpp
example_cuda.cu
__init__.py
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

For linear algebra rely on optimized libraries like OpenBLAS or MKL. Pybind11 can expose these routines to Python. Example: `cblas_sgemm` for matrix multiplication.

### Full Pybind example

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <immintrin.h>
#ifdef _OPENMP
#include <omp.h>
#endif
namespace py = pybind11

py::array_t<float> add_vectorized(py::array_t<float> a py::array_t<float> b) {
    auto buf_a = a.request() 
    auto buf_b = b.request() 
    if(buf_a.size != buf_b.size)
        throw std::runtime_error("Arrays must be same size")

    auto result = py::array_t<float>(buf_a.size)
    auto buf_r = result.request()
    float* ptr_a = (float*)buf_a.ptr
    float* ptr_b = (float*)buf_b.ptr
    float* ptr_r = (float*)buf_r.ptr
    size_t n = buf_a.size

    #pragma omp parallel for
    for(size_t i = 0 i < n i += 8) {
        __m256 va = _mm256_loadu_ps(ptr_a + i)
        __m256 vb = _mm256_loadu_ps(ptr_b + i)
        __m256 vr = _mm256_add_ps(va vb)
        _mm256_storeu_ps(ptr_r + i vr)
    }
    return result
}

PYBIND11_MODULE(example m) {
    m.def("add_vectorized" &add_vectorized "Vectorized float addition with SIMD and OpenMP")
}
```

---

## 5 CUDA GPU function

example_cuda.cu

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>
namespace py = pybind11

__global__ void add_kernel(float* a float* b float* out int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x
    if(idx < n) out[idx] = a[idx] + b[idx]
}

py::array_t<float> add_cuda(py::array_t<float> a py::array_t<float> b) {
    auto buf_a = a.request()
    auto buf_b = b.request()
    if(buf_a.size != buf_b.size)
        throw std::runtime_error("Arrays must be same size")
    int n = buf_a.size

    float *d_a *d_b *d_out
    cudaMalloc(&d_a n*sizeof(float))
    cudaMalloc(&d_b n*sizeof(float))
    cudaMalloc(&d_out n*sizeof(float))
    cudaMemcpy(d_a buf_a.ptr n*sizeof(float) cudaMemcpyHostToDevice)
    cudaMemcpy(d_b buf_b.ptr n*sizeof(float) cudaMemcpyHostToDevice)

    int block = 256 grid = (n + block - 1)/block
    add_kernel<<<grid block>>>(d_a d_b d_out n)

    auto result = py::array_t<float>(n)
    cudaMemcpy(result.mutable_data() d_out n*sizeof(float) cudaMemcpyDeviceToHost)
    cudaFree(d_a) cudaFree(d_b) cudaFree(d_out)
    return result
}

PYBIND11_MODULE(example m) {
    m.def("add_cuda" &add_cuda "Add arrays on GPU with CUDA")
}
```

Compile with:

```bash
nvcc --compiler-options '-fPIC' -O3 -shared example_cuda.cu -o example_cuda.so -lcudart
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
python setup.py bdist_wheel
pip install dist/example-0.1.0-cp311-cp311-linux_x86_64.whl
```

Key difference: `.so` is the **runtime artifact**. Wheel is the **distribution format** containing `.so` files, metadata, and Python package structure.

---

## 7 Setup script

setup.py

```python
from setuptools import setup, Extension
import pybind11
import os
import sys
from setuptools.command.build_ext import build_ext
import subprocess

# Custom class to compile CUDA with nvcc
class BuildExtWithCUDA(build_ext):
    def build_extensions(self):
        for ext in self.extensions:
            if getattr(ext, 'cuda', False):
                self.build_cuda_extension(ext)
            else:
                super().build_extensions()
    
    def build_cuda_extension(self, ext):
        # Build the CUDA .cu file into a shared library
        output_dir = os.path.abspath(self.build_lib)
        lib_file = os.path.join(output_dir, ext.name + ".so")
        os.makedirs(output_dir, exist_ok=True)
        cmd = [
            "nvcc",
            "-O3",
            "--compiler-options", "'-fPIC'",
            "-shared",
            *ext.sources,
            "-o", lib_file,
            "-Xcompiler", "-fPIC"
        ]
        print("Building CUDA extension:", " ".join(cmd))
        subprocess.check_call(" ".join(cmd), shell=True)

# CPU vectorized extension
cpu_ext = Extension(
    "example",
    ["example.cpp"],
    include_dirs=[pybind11.get_include()],
    extra_compile_args=["-O3", "-march=native", "-fopenmp"],
    language="c++"
)

# CUDA extension
cuda_ext = Extension(
    "example_cuda",
    ["example_cuda.cu"],
    include_dirs=[pybind11.get_include()],
)
cuda_ext.cuda = True  # mark it as CUDA

setup(
    name="example",
    ext_modules=[cpu_ext, cuda_ext],
    cmdclass={"build_ext": BuildExtWithCUDA},
    zip_safe=False,
)

```

---

## 8 Demo and Benchmark

```python
import example
import numpy as np
import time

# Simple sanity check
print("Simple addition:", example.add_simple(1, 2))

# Prepare large arrays for benchmarking
N = 10_000_000
a = np.random.rand(N).astype(np.float32)
b = np.random.rand(N).astype(np.float32)

# Vectorized CPU addition
start_cpu = time.time()
c_cpu = example.add_vectorized(a, b)
end_cpu = time.time()
print(f"CPU vectorized addition first 10 elements: {c_cpu[:10]}")
print(f"CPU vectorized addition time: {end_cpu - start_cpu:.6f} seconds")

# CUDA GPU addition
start_gpu = time.time()
c_gpu = example.add_cuda(a, b)
end_gpu = time.time()
print(f"GPU CUDA addition first 10 elements: {c_gpu[:10]}")
print(f"GPU CUDA addition time: {end_gpu - start_gpu:.6f} seconds")

# Verify correctness
if np.allclose(c_cpu, c_gpu):
    print("CPU and GPU results match!")
else:
    print("Warning: CPU and GPU results do not match!")
```
