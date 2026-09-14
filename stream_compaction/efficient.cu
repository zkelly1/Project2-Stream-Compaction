#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        Common::PerformanceTimer& timer() {
            static Common::PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpsweep(int pairs, int width, int *data) {
            // Each thread handles one pair of subtree results.
            int pair = blockIdx.x * blockDim.x + threadIdx.x;

            if (pair < pairs) {
                // width is the number of elements covered by this pair.
                // For width 4, the first pair uses positions 1 and 3.
                int right = (pair + 1) * width - 1;
                int left = right - width / 2;

                // Keep the combined sum at the right end of this subtree.
                data[right] += data[left];
            }
        }

        __global__ void kernClearRoot(int n, int *data) {
            // The root starts with the sum BEFORE the whole array.
            // Nothing comes before the array, so this prefix is zero.
            data[n - 1] = 0;
        }

        __global__ void kernDownsweep(int pairs, int width, int *data) {
            // Each thread handles one pair of subtree results.
            int pair = blockIdx.x * blockDim.x + threadIdx.x;

            if (pair < pairs) {
                // width is the number of elements covered by this pair.
                // For width 4, the first pair uses positions 1 and 3.
                int right = (pair + 1) * width - 1;
                int left = right - width / 2;

                // Save the left sum before replacing it with a prefix.
                int leftSum = data[left];

                // The left half gets the prefix of the whole subtree.
                data[left] = data[right];

                // The right half also needs the sum of the left half,
                // since those values come before it.
                data[right] += leftSum;
            }
        }

        // Recitation slides 6-7: sum up the tree, then pass prefixes back down.
        // One thread handles one pair. The grid shrinks with the number of pairs.
        void scanDevice(int padded, int *data) {
            int blockSize = Common::blockSize;
            int levels = ilog2(padded);

            // First combine small groups into bigger groups.
            // Each new launch waits for the previous level in the same stream.
            for (int level = 1; level <= levels; ++level) {
                int width = 1 << level;
                // Wider groups mean fewer pairs for threads to work on.
                int pairs = padded / width;

                kernUpsweep<<<(pairs + blockSize - 1) / blockSize, blockSize>>>(pairs, width, data);
                checkCUDAError("upsweep");
            }

            // Replace the total sum before we start passing prefixes down.
            kernClearRoot<<<1, 1>>>(padded, data);
            checkCUDAError("clear root");

            // Now walk the tree in reverse, down to individual elements.
            for (int level = levels; level >= 1; --level) {
                int width = 1 << level;
                // Wider groups mean fewer pairs for threads to work on.
                int pairs = padded / width;

                kernDownsweep<<<(pairs + blockSize - 1) / blockSize, blockSize>>>(pairs, width, data);
                checkCUDAError("downsweep");
            }
        }

        void scan(int n, int *odata, const int *idata) {
            // Handle an empty array without allocating or reading anything.
            if (n <= 0) {
                timer().startGpuTimer();
                timer().endGpuTimer();
                return;
            }

            // The tree needs a power of two. For example, 5 becomes 8.
            int padded = 1 << ilog2ceil(n);

            // Fill the extra positions with zero so they add nothing.
            // Only n input values are copied from the CPU.
            int *dev_data;
            cudaMalloc(&dev_data, padded * sizeof(int));
            cudaMemset(dev_data, 0, padded * sizeof(int));
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("efficient setup");

            // Time the tree passes, not allocation or host/device copies.
            timer().startGpuTimer();
            scanDevice(padded, dev_data);
            timer().endGpuTimer();
            checkCUDAError("efficient scan");

            // The padded positions were only needed for the tree.
            // Copy back just the original number of elements.
            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_data);
            checkCUDAError("efficient cleanup");
        }

        int compact(int n, int *odata, const int *idata) {
            // Handle an empty array without allocating or reading anything.
            if (n <= 0) {
                timer().startGpuTimer();
                timer().endGpuTimer();
                return 0;
            }

            // The tree needs a power of two. For example, 5 becomes 8.
            int padded = 1 << ilog2ceil(n);

            // Keep the original values, output, flags, and prefixes separate.
            // Only the prefix array needs room for the padded tree.
            int *dev_input, *dev_output, *dev_keep, *dev_indices;
            cudaMalloc(&dev_input, n * sizeof(int));
            cudaMalloc(&dev_output, n * sizeof(int));
            cudaMalloc(&dev_keep, n * sizeof(int));
            cudaMalloc(&dev_indices, padded * sizeof(int));
            cudaMemset(dev_indices, 0, padded * sizeof(int));
            cudaMemcpy(dev_input, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("compact setup");

            // Round up to cover every real input element.
            int blockSize = Common::blockSize;
            int blocks = (n + blockSize - 1) / blockSize;

            timer().startGpuTimer();

            // First mark which values are non zero in both arrays.
            // We keep one copy because the scan changes the other copy.
            Common::kernMapToBoolean<<<blocks, blockSize>>>(n, dev_keep, dev_input);
            Common::kernMapToBoolean<<<blocks, blockSize>>>(n, dev_indices, dev_input);
            checkCUDAError("compact map");

            // The exclusive prefix counts kept values BEFORE each element.
            // That count becomes its output position.
            scanDevice(padded, dev_indices);

            // Write only kept values into their assigned positions.
            Common::kernScatter<<<blocks, blockSize>>>(n, dev_output, dev_input, dev_keep, dev_indices);
            checkCUDAError("compact scatter");
            timer().endGpuTimer();
            checkCUDAError("compact scan");

            // The last prefix does not include the last element itself.
            // Add its flag to get the full number of kept elements.
            int count, lastKeep;
            cudaMemcpy(&count, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastKeep, dev_keep + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            count += lastKeep;

            // Only the first count output positions contain valid results.
            if (count > 0) {
                cudaMemcpy(odata, dev_output, count * sizeof(int), cudaMemcpyDeviceToHost);
            }

            // We are done with the temporary GPU arrays.
            cudaFree(dev_input);
            cudaFree(dev_output);
            cudaFree(dev_keep);
            cudaFree(dev_indices);
            checkCUDAError("compact cleanup");

            return count;
        }
    }
}
