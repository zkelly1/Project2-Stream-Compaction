#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
    namespace Thrust {
        Common::PerformanceTimer& timer() {
            static Common::PerformanceTimer timer;
            return timer;
        }

        void scan(int n, int *odata, const int *idata) {
            // Avoid creating an input range from an empty array.
            if (n <= 0) {
                timer().startGpuTimer();
                timer().endGpuTimer();
                return;
            }

            // These vectors own their GPU memory and free it automatically.
            // Create them before timing so setup is not measured.
            thrust::device_vector<int> input(idata, idata + n);
            thrust::device_vector<int> output(n);

            // Thrust performs the same exclusive sum as our own scans.
            timer().startGpuTimer();
            thrust::exclusive_scan(input.begin(), input.end(), output.begin());
            timer().endGpuTimer();
            checkCUDAError("thrust scan");

            // Bring the answer back to the CPU after timing the scan.
            thrust::copy(output.begin(), output.end(), odata);
        }
    }
}
