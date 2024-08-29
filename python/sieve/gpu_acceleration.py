import cupy as cp

GPU_THREADS_PER_BLOCK = 128


def initialize(sieve):
    # Free up any memory blocks that are currently in use
    cp.cuda.MemoryPool().free_all_blocks()

    # Copy the sieve to the GPU
    return cp.array(sieve)


def launch_kernel(sieve, prime, upper_bound):
    # Calculate the number of blocks per grid
    blocks_per_grid = (upper_bound - prime * prime + GPU_THREADS_PER_BLOCK - 1)

    # Launch the CUDA kernel
    mark_multiples_gpu((blocks_per_grid,),
                       (GPU_THREADS_PER_BLOCK,),
                       (sieve, prime, upper_bound))

    # Synchronize the device to ensure completion
    cp.cuda.Device().synchronize()


kernel_code = '''
extern "C" __global__
void mark_multiples_gpu(bool* sieve, int prime, int upper_bound) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int multiple = prime * prime + prime * 2 * idx;
    if (multiple <= upper_bound && idx < upper_bound) {
        sieve[multiple / 2] = false;
    }
}
'''

mark_multiples_gpu = cp.RawKernel(kernel_code, 'mark_multiples_gpu')
