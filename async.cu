#include <cstdio>
#include <cstdlib>
#include <stdint.h>
#include <vector>
#include <assert.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
  do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
      exit(1); \
    } \
  } while (0)

template <typename T>
__global__ void pipeline_kernel_sync(T *global, uint64_t *clock, size_t copy_count) {
  extern __shared__ char s[];
  T *shared = reinterpret_cast<T *>(s);

  uint64_t clock_start = clock64();

  for (size_t i = 0; i < copy_count; ++i) {
    shared[blockDim.x * i + threadIdx.x] = global[blockDim.x * i + threadIdx.x];
  }

  uint64_t clock_end = clock64();

  atomicAdd(reinterpret_cast<unsigned long long *>(clock), (unsigned long long)(clock_end - clock_start));
}

template <typename T>
__global__ void pipeline_kernel_async(T *global, uint64_t *clock, size_t copy_count) {
  extern __shared__ char s[];
  T *shared = reinterpret_cast<T *>(s);

  uint64_t clock_start = clock64();

  for (size_t i = 0; i < copy_count; ++i) {
    __pipeline_memcpy_async(&shared[blockDim.x * i + threadIdx.x],
                            &global[blockDim.x * i + threadIdx.x], sizeof(T));
  }
  __pipeline_commit();
  __pipeline_wait_prior(0);

  uint64_t clock_end = clock64();

  atomicAdd(reinterpret_cast<unsigned long long *>(clock), (unsigned long long)(clock_end - clock_start));
}

// A helper function to run either the sync or async kernel
template<typename T>
double run_test(bool async_mode, size_t num_elements, int block_size) {
  // num_elements = copy_count * blockDim.x
  // Decide copy_count from num_elements and block_size
  size_t copy_count = num_elements / block_size;
  assert(copy_count * block_size == num_elements);

  // Allocate device memory
  T *d_global;
  uint64_t *d_clock;
  CHECK_CUDA(cudaMalloc(&d_global, num_elements * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_clock, sizeof(uint64_t)));

  // Initialize d_global with some data
  std::vector<T> h_data(num_elements, (T)1);
  CHECK_CUDA(cudaMemcpy(d_global, h_data.data(), num_elements * sizeof(T), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_clock, 0, sizeof(uint64_t)));

  // Kernel configuration
  dim3 block(block_size);
  dim3 grid(1); // single block for simplicity
  size_t shared_mem_size = num_elements * sizeof(T);

  // Launch kernel
  if (!async_mode) {
    pipeline_kernel_sync<T><<<grid, block, shared_mem_size>>>(d_global, d_clock, copy_count);
  } else {
    pipeline_kernel_async<T><<<grid, block, shared_mem_size>>>(d_global, d_clock, copy_count);
  }
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  // Read clock results
  uint64_t h_clock = 0;
  CHECK_CUDA(cudaMemcpy(&h_clock, d_clock, sizeof(uint64_t), cudaMemcpyDeviceToHost));

  // Clean up
  CHECK_CUDA(cudaFree(d_global));
  CHECK_CUDA(cudaFree(d_clock));

  // h_clock now holds total clock cycles for the kernel launch over 1 block
  // To get average per thread, divide by (block_size).
  // For comparing results, we may just use total cycles as is, or average per element/thread.
  double avg_clock_cycles = (double)h_clock / (double)block_size;
  return avg_clock_cycles;
}

int main(int argc, char *argv[]) {
  // Set up some test parameters:
  // We'll vary the total number of bytes copied by changing num_elements.
  // We'll fix the block_size and run for multiple element sizes and modes.
  
  // For demonstration: 
  // block_size = 128 threads, vary total bytes from say 512 to 48,640 as in the figure
  // We'll try element sizes = 4,8,16 bytes (float/int, double2, etc.)
  // We'll do sync and async tests.
  
  int block_size = 128;
  // Range of bytes copied:
  // For example: from 512 bytes up to ~48k bytes
  // Make sure it's divisible by element_size and block_size.
  
  // We'll pick a series of test sizes, ensuring divisibility:
  std::vector<size_t> test_bytes = {512, 1536, 2256, 3584, 4608, 6656, 7680,
                                    8704, 10752, 11776, 12800, 13824, 14848, 
                                    15872, 16896, 17920, 18944, 19968, 20992, 
                                    22016, 23040, 24064, 25088, 26112, 27136, 
                                    28160, 29184, 30208, 31232, 32256, 33280,
                                    34304, 35328, 36352, 37376, 38400, 39424,
                                    40448, 41472, 42496, 43520, 44544, 45568,
                                    46592, 47616, 48640};

  // We'll run tests for T = 4-byte type, 8-byte type, 16-byte type
  // For simplicity let's define types:
  struct type4 { int x; };   // 4 bytes
  struct type8 { int2 x; };  // 8 bytes
  struct type16 { int4 x; }; // 16 bytes
  
  // Print header
  // Format: bytes, sync(elem_size=4), sync(elem_size=8), sync(elem_size=16),
  //         async(elem_size=4), async(elem_size=8), async(elem_size=16)
  printf("Bytes,Sync(4),Sync(8),Sync(16),Async(4),Async(8),Async(16)\n");

  for (auto bytes : test_bytes) {
    // Compute num_elements for each type
    // Ensure that bytes is divisible by element_size * block_size
    // If not divisible, skip.
    if ((bytes % (sizeof(type4)*block_size)) != 0) continue;
    if ((bytes % (sizeof(type8)*block_size)) != 0) continue;
    if ((bytes % (sizeof(type16)*block_size)) != 0) continue;

    size_t elements_4 = bytes / sizeof(type4);
    size_t elements_8 = bytes / sizeof(type8);
    size_t elements_16 = bytes / sizeof(type16);

    double sync4   = run_test<type4>(false, elements_4, block_size);
    double sync8   = run_test<type8>(false, elements_8, block_size);
    double sync16  = run_test<type16>(false, elements_16, block_size);

    double async4  = run_test<type4>(true, elements_4, block_size);
    double async8  = run_test<type8>(true, elements_8, block_size);
    double async16 = run_test<type16>(true, elements_16, block_size);

    printf("%zu,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n",
           bytes, sync4, sync8, sync16, async4, async8, async16);
  }

  return 0;
}
