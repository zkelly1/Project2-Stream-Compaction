#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        StreamCompaction::Common::PerformanceTimer& timer();

        // Read n CPU values and write their exclusive prefix sums to odata.
        void scan(int n, int *odata, const int *idata);

        // Return the number kept. Only that many output values are valid.
        int compactWithoutScan(int n, int *odata, const int *idata);

        // Return the number kept. Only that many output values are valid.
        int compactWithScan(int n, int *odata, const int *idata);
    }
}
