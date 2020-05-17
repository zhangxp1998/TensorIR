# TensorIR

## Overview

TensorIR is a scala library that allows you to train a Neural Network with relatively few lines of code. It will automatically generate efficient C++ code, optimize it (for now there's only memory planning optimization), compile it, and run it.



## Repository Structure

* `src` : Contains scala code responsible for generating C++ code
  * `src/scala/tensor/ir/` contains frontend code that creates IR nodes
    * `src/scala/tensor/ir/CPUTensorOps` defines basic tensor operations: Plus/Sub/Multiply/Divide, convolution, batchnorm, etc.
    * `src/scala/tensor/ir/CPUTensorDiff` defines Auto-Diff versions of the same operations.
    * `src/scala/tensor/ir/ResNet` contains a small example neural network built with the current IR. It currently runs on CPU backend, to use GPU backend, change `val dslDriver = new CPUTensorDiffDriverC[String,Unit]` to `val dslDriver = new GPUTensorDiffDriverC[String,Unit]` . Simply switching the driver used is sufficient.
  * `src/scala/tensor/ir/backend` contains backend code that generates C++ (or CUDA) code from IR nodes created by the frontend.
  * `src/scala/tensor/ir/backend/MemoryAnalysis.scala` is responsible for extracting tensor lifetime information. (when is a tensor allocated, when can it be freed.) It returns a `Map[Int, MemoryEvent]` , when the integer represents an arbitrary timestamp, `MemoryEvent` is an event that signals either beginning of end of a tensor's lifetime.
  * `src/scala/tensor/ir/StagedMemoryAllocstor` is responsible to taking in tensor lifetime information, and emit a feasible memory plan. It uses a simple best-fit strategy. `MemorySolver` in the same directory uses z3, but is is too slow.
  * `src/scala/tensor/ir/backend/CPUMemoryPlanningTransformer` is responsible to taking in a memory plan(emited by `StagedMemoryAllocstor` or `MemorySolver` ) and an IR graph, and returning an modified IR graph with the specified memory plan deployed.
* `gen` Contains build definition files for generated C++(or CUDA) code, also contains runtime libraries for generated code. Currently, `CMake` is used to build the generated code.
* `lms-clean` is a submodule of the Light Weight Modular Staging framework.
  * `TensorIR` uses a [fork](https://github.com/zhangxp1998/lms-clean) of the LMS framework. This fork has 2 important modifications:
    * Prevent inlining of some tensor operations to preserve lifetime information of tensors.
    * Use `CMake` to build generated source code, instead of manually synthesizing compile commands
* `test` contains a few Unit testcases for CPU backend. 

## Dependencies

The CPU backend relies on intel's [MKL-dnn](https://github.com/oneapi-src/oneDNN)(installable by `brew install mkl-dnn` on mac), the GPU backend relies on CUDA, cuDNN, and thrust.