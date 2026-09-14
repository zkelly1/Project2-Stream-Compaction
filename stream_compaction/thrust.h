#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Thrust {
        StreamCompaction::Common::PerformanceTimer& timer();

        // Read n CPU values and write their exclusive prefix sums to odata.
        void scan(int n, int *odata, const int *idata);
    }
}
