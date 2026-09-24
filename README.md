# MIGScope
This repository contains the implementation of the GPU TLB side-channel attack
used in our work.

The implementation targets NVIDIA A100 GPUs with CUDA support.

## Requirements
### Software

- Ubuntu 20.04 (recommended)
- NVIDIA Driver >= 535
- CUDA Toolkit >= 12.0


## Compilation

Compile the CUDA program using:

```bash
nvcc \
-O0 \
-Xptxas -dscm=wt \
-Xptxas -dlcm=cg \
-arch=sm_80 \
scan.cu \
-o scan

```
## Running

Before running the program, make sure that the GPU environment is properly
configured and the target GPU is available.
Execute the compiled binary:

./scan
