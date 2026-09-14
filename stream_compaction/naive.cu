#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        Common::PerformanceTimer& timer() {
            // We reuse the same timer for each call.
            static Common::PerformanceTimer timer;
            return timer;
        }

        __global__ void kernShift(int n, int *output, const int *input) {
            // Give each thread one position in the full array.
            int index = blockIdx.x * blockDim.x + threadIdx.x;

            if (index < n) {
                // Move each value one position to the right.
                // The first position gets zero because nothing comes before it.
                output[index] = index == 0 ? 0 : input[index - 1];
            }
        }

        __global__ void kernAddPrevious(int n, int distance, int *output, const int *input) {
            // Give each thread one position in the full array.
            int index = blockIdx.x * blockDim.x + threadIdx.x;

            if (index < n) {
                int sum = input[index];

                // Only add the earlier value if that position exists.
                // The first few threads just keep their current sum.
                if (index >= distance) {
                    sum += input[index - distance];
                }

                // Write to the OTHER buffer so we do not change
                // a value that another thread still needs to read.
                output[index] = sum;
            }
        }

        void scan(int n, int *odata, const int *idata) {
            // There is no data to scan for an empty input.
            if (n <= 0) {
                timer().startGpuTimer();
                timer().endGpuTimer();
                return;
            }

            // Allocate both GPU arrays and copy over the input.
            // This setup is outside the algorithm timer.
            int *dev_read, *dev_write;
            cudaMalloc(&dev_read, n * sizeof(int));
            cudaMalloc(&dev_write, n * sizeof(int));
            cudaMemcpy(dev_read, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("naive setup");

            // Round up so the last partial block is included too.
            int blockSize = Common::blockSize;
            int blocks = (n + blockSize - 1) / blockSize;

            timer().startGpuTimer();

            // First shift the input, as in lecture slide 20.
            // The additions will then produce an EXCLUSIVE scan.
            kernShift<<<blocks, blockSize>>>(n, dev_write, dev_read);
            checkCUDAError("naive shift");
            std::swap(dev_read, dev_write);

            // Look back 1, 2, 4, 8, ... positions on successive passes.
            // Each pass combines sums from the previous pass.
            for (int level = 0; level < ilog2ceil(n); ++level) {
                kernAddPrevious<<<blocks, blockSize>>>(n, 1 << level, dev_write, dev_read);
                checkCUDAError("naive pass");

                // The output of this pass becomes the input of the next.
                // Swapping pointers does not copy the arrays.
                std::swap(dev_read, dev_write);
            }

            timer().endGpuTimer();
            checkCUDAError("naive scan");

            // dev_read holds the final result after the last swap.
            // Copy it back and release both GPU arrays after timing.
            cudaMemcpy(odata, dev_read, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_read);
            cudaFree(dev_write);
            checkCUDAError("naive cleanup");
        }
    }
}
