# Sieve of Eratosthenes

## Overview
This project is designed to compute prime numbers efficiently using the Sieve
of Eratosthenes algorithm. The implementation also optionally leverages GPU
acceleration with CuPy to optimize performance.

### Key Features
- **GPU Acceleration:** Utilizes CuPy to offload computations to the GPU,
improving performance for large-scale prime number generation.
- **Optimized Sieve Size:** By focusing exclusively on odd numbers, the
algorithm halves the size of the sieve, reducing memory consumption and
computational complexity.
- **Square-Root Limitation:** The algorithm smartly limits calculations to the
square of the current prime, eliminating unnecessary operations and enhancing
efficiency.
- **Prime Number Theorem Integration:** The sieve's upper bound is dynamically
determined based on the Prime Number Theorem, ensuring that only the highest
likely primes are considered, further optimizing the computational workload.


## Prerequisites
- Python 3.8+

For running tests with GPU acceleration:
- NVIDIA GPU with CUDA support 
- CUDA Toolkit (version 11.x or higher)

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/kalebpomeroy/bhe-code-exercise.git 
cd bhe-code-exercise
```

### 2. Set up a virtual environment and install dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running Tests
Once the environmennt is setup, you can run tests like below 
```bash
python -m unittest test_sieve.py
python -m unittest test_sieve_gpu.py
```

Note: Some tests for the GPU are commented out as GPU memory and 
load is a very limiting factor and hardware dependant. There may be 
some configuration and performance tweaking necessary for larger primes
(see gpu_acceleration.py for that)