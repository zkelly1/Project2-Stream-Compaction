#include <cstdio>
#include <vector>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();

            int sum = 0;
            for (int i = 0; i < n; ++i) {
                // We save the current sum into the output
                // data first before changing it. This is
                // how / why it is EXCLUSIVE
                odata[i] = sum;

                int value = idata[i];
                sum += value;
            }

            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int count = 0;

            // Here, we just add non zero
            // elements to an array, very simple
            for (int i = 0; i < n; ++i) {
                if (idata[i] != 0) {
                    odata[count++] = idata[i];
                }
            }
            timer().endCpuTimer();
            return count;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            // Initialize our arrays
            // that tell us what indices to keep on the first past (non zero)
            std::vector<int> keep(n), indices(n);

            timer().startCpuTimer();

            // First go over and find the non zero
            // indices. This populates the keep array
            for (int i = 0; i < n; ++i) {
                keep[i] = idata[i] != 0;
            }

            timer().endCpuTimer();
            float mapTime = timer().getCpuElapsedTimeForPreviousOperation();

            // Scan the keep flags not the original values.
            // Each prefix tells us how many kept values come before it.
            scan(n, indices.data(), keep.data());
            float scanTime = timer().getCpuElapsedTimeForPreviousOperation();

            timer().startCpuTimer();

            // Put each kept value at the position given by its prefix.
            for (int i = 0; i < n; ++i) {
                if (keep[i]) {
                    odata[indices[i]] = idata[i];
                }
            }

            // The last prefix excludes the last flag, so add that flag back.
            // For an empty input, there are no flags to read.
            int count = n > 0 ? indices[n - 1] + keep[n - 1] : 0;
            timer().endCpuTimer();

            // scan uses this same timer. Add all three stage times together
            // so the reported time covers map, scan, and scatter.
            timer().addCpuElapsedTime(mapTime + scanTime);
            return count;
        }
    }
}
