# Physical Signal Anomaly Detection

A personal research project connecting physical datasets, machine learning, mechanistic interpretability, Rust, and GPU computing.

**Status: project definition.** This repository currently contains the project brief. Data collection, implementation, experiments, and benchmarks are planned work.

## Research question

Can a small model detect unusual behavior in the sound or vibration of a physical system, and can we explain which signal properties and internal model mechanisms drive its decisions?

The first experiment will use one signal type and one physical setup. The object, sensor, and recording protocol will be selected before data collection.

## Intended outputs

* A documented dataset with recording metadata, labels, quality checks, and clearly separated training and evaluation sessions.
* A reproducible CPU baseline and a small anomaly detection model.
* Interpretability experiments that test hypotheses about what the model has learned.
* A measured comparison of selected CPU and GPU computations, if profiling identifies a useful candidate.
* Clear instructions for reproducing the results, including unsuccessful experiments and limitations.

## How the three research areas connect

| Area | Contribution |
| --- | --- |
| Datasets | Collect, clean, annotate, and document physical recordings. Keep provenance, sampling rate, units, device, and session information. |
| Interpretability | Inspect a small neural model and intervene on selected internal units or representations to test their effect on anomaly scores. Signal plots alone will not be presented as mechanistic explanations. |
| AI systems and HPC | Build a Rust signal processing component, profile the workload, and investigate a PTX implementation of one suitable GPU operation. Inspect generated SASS when it helps explain performance. |

## Where Rust and PTX could help

One possible target is computing the root mean square (RMS) of many signal windows:

`rms(window) = sqrt(sum(sample * sample) / window_length)`

RMS measures signal magnitude. It is an example preprocessing feature, not a complete anomaly detector.

1. Establish a trusted CPU reference and measure its runtime.
2. Implement the same operation in Rust on the CPU.
3. If the workload is large enough to justify the experiment, implement the window reduction in PTX and launch it from Rust through the CUDA driver.
4. Compare numerical results within a stated tolerance, then measure both kernel time and total time including allocation, transfers, launch, synchronization, and result retrieval.
5. Report which batch sizes benefit, if any. A small recording may be faster on the CPU because GPU overhead can outweigh the computation saved.

Rust manages the application and GPU resources. PTX specifies the device computation. NVIDIA tools compile PTX into native GPU instructions, represented as SASS. The Rust compiler does not prove the correctness or memory safety of a handwritten PTX kernel.

The performance question is whether a measured part of the pipeline benefits from GPU execution. No acceleration result is claimed in advance, and replacing mature NVIDIA libraries is not a project requirement.

## Planned milestones

1. Choose the physical setup and define normal and anomalous conditions.
2. Collect or select data and establish quality checks. Split recordings by session or device as appropriate before creating overlapping windows, to reduce leakage.
3. Build and evaluate a simple CPU baseline. Select evaluation metrics and a decision threshold without using the held out test recordings.
4. Train a small model and test interpretability hypotheses.
5. Profile the pipeline and decide whether to run the Rust and PTX comparison.
6. Publish reproducible experiments, measurements, and conclusions.

Only the project definition is complete. Each implementation milestone will be scoped and reviewed before the next one starts.

## References

* [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
* [NVIDIA CUDA Driver API: module loading](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__MODULE.html)
* [cudarc: CUDA driver integration in Rust](https://docs.rs/cudarc/latest/cudarc/driver/index.html)
