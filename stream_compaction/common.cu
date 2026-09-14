#include "common.h"

void checkCUDAErrorFn(const char *msg, const char *file, int line) {
    // Stop on a CUDA error instead of continuing with an invalid result.
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
}

namespace StreamCompaction {
    namespace Common {
        int blockSize = 128;

        /**
         * Maps an array to an array of 0s and 1s for stream compaction. Elements
         * which map to 0 will be removed, and elements which map to 1 will be kept.
         */
        __global__ void kernMapToBoolean(int n, int *bools, const int *idata) {
            // Convert the thread position in its block to an array index.
            int index = blockIdx.x * blockDim.x + threadIdx.x;

            // The last block can contain threads past the end of the array.
            if (index < n) {
                // A non zero value gets 1, meaning we want to keep it.
                bools[index] = idata[index] != 0;
            }
        }

        /**
         * Performs scatter on an array. That is, for each element in idata,
         * if bools[idx] == 1, it copies idata[idx] to odata[indices[idx]].
         */
        __global__ void kernScatter(int n, int *odata,
                const int *idata, const int *bools, const int *indices) {
            // Convert the thread position in its block to an array index.
            int index = blockIdx.x * blockDim.x + threadIdx.x;

            // Skip removed values. Each kept value has a unique prefix,
            // so no two kept values write to the same output position.
            if (index < n && bools[index]) {
                odata[indices[index]] = idata[index];
            }
        }

    }
}
