#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <stream_compaction/cpu.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/thrust.h>

using namespace StreamCompaction;

float run(const std::string &mode, int n, int *output, const int *input, int &count) {
    // Scans return n values. Compaction changes count to the number kept.
    count = n;

    if (mode == "cpu") {
        CPU::scan(n, output, input);
        return CPU::timer().getCpuElapsedTimeForPreviousOperation();
    }

    if (mode == "naive") {
        Naive::scan(n, output, input);
        return Naive::timer().getGpuElapsedTimeForPreviousOperation();
    }

    if (mode == "efficient") {
        Efficient::scan(n, output, input);
        return Efficient::timer().getGpuElapsedTimeForPreviousOperation();
    }

    if (mode == "thrust") {
        Thrust::scan(n, output, input);
        return Thrust::timer().getGpuElapsedTimeForPreviousOperation();
    }

    if (mode == "cpu-compact") {
        count = CPU::compactWithoutScan(n, output, input);
        return CPU::timer().getCpuElapsedTimeForPreviousOperation();
    }

    if (mode == "cpu-scan-compact") {
        count = CPU::compactWithScan(n, output, input);
        return CPU::timer().getCpuElapsedTimeForPreviousOperation();
    }

    if (mode == "gpu-compact") {
        count = Efficient::compact(n, output, input);
        return Efficient::timer().getGpuElapsedTimeForPreviousOperation();
    }
    throw std::runtime_error("Unknown implementation: " + mode);
}

bool isCompact(const std::string &mode) {
    return mode.find("compact") != std::string::npos;
}

std::vector<int> expectedResult(const std::vector<int> &input, bool compact) {
    // Use the standard library to build the answer independently.
    // A bug in our CPU scan should not make the GPU tests pass by mistake.
    std::vector<int> expected;

    if (compact) {
        std::copy_if(input.begin(), input.end(), std::back_inserter(expected),
            [](int value) { return value != 0; });
    } else {
        expected.resize(input.size());
        std::exclusive_scan(input.begin(), input.end(), expected.begin(), 0);
    }
    return expected;
}

void verify(const std::string &mode, const std::vector<int> &input) {
    // Put a guard value on either side of the writable output.
    // The implementation gets a pointer just after the first guard.
    const int sentinel = 123456789;
    std::vector<int> output(input.size() + 2, sentinel);
    std::vector<int> expected = expectedResult(input, isCompact(mode));

    int count;
    run(mode, static_cast<int>(input.size()), output.data() + 1, input.data(), count);

    // Check the count, the values, and both guards.
    if (count != static_cast<int>(expected.size()) ||
        !std::equal(expected.begin(), expected.end(), output.begin() + 1) ||
        output.front() != sentinel || output.back() != sentinel) {
        throw std::runtime_error(mode + " failed for size " + std::to_string(input.size()));
    }
}

void tests() {
    std::mt19937 random(5650);
    const std::vector<std::string> modes = {
        "cpu", "naive", "efficient", "thrust", "cpu-compact", "cpu-scan-compact", "gpu-compact"
    };

    // Include empty inputs and lengths around block and power-of-two boundaries.
    const std::vector<int> sizes = {
        0, 1, 2, 3, 7, 13, 31, 32, 33, 37, 123, 127, 128, 129,
        253, 256, 257, 457, 1003, 1023, 1024, 1025, 8192, 10000, 65536, 1000000, 1048576
    };

    // Try every implementation with the same collection of test cases.
    for (const auto &mode : modes) {
        int cases = 0;
        for (int blockSize : {32, 128, 1024}) {
            Common::blockSize = blockSize;

            // Check the small lecture example before the generated arrays.
            verify(mode, {3, 1, 7, 0, 4, 1, 6, 3});
            ++cases;

            for (int n : sizes) {
                for (int pattern = 0; pattern < 5; ++pattern) {
                    std::vector<int> input(n);

                    // Test zeros, ones, alternating negatives, random values,
                    // and a single kept value at the very end.
                    for (int i = 0; i < n; ++i) {
                        if (pattern == 0) {
                            input[i] = 0;
                        }
                        if (pattern == 1) {
                            input[i] = 1;
                        }
                        if (pattern == 2) {
                            input[i] = i % 2 == 0 ? 0 : -3;
                        }
                        if (pattern == 3) {
                            input[i] = static_cast<int>(random() % 11) - 5;
                        }
                        if (pattern == 4) {
                            input[i] = i == n - 1 ? 9 : 0;
                        }
                    }

                    verify(mode, input);
                    ++cases;
                }
            }
        }
        printf("PASS %-17s %d cases\n", mode.c_str(), cases);
    }
    printf("All 2856 checks passed (27 sizes, 5 patterns, 3 block sizes, lecture example).\n");
    printf("Example input:   [3 1 7 0 4 1 6 3]\n");
    printf("Exclusive scan: [0 3 4 11 11 15 16 22]\n");
    printf("Compaction:     [3 1 7 4 1 6 3]\n");
}

int main(int argc, char **argv) {
    try {
        // With no arguments, just run the correctness tests.
        if (argc == 1 || (argc == 2 && std::string(argv[1]) == "--test")) {
            tests();
            return 0;
        }

        // Benchmark mode needs all six settings after its flag.
        if (argc != 8 || std::string(argv[1]) != "--benchmark") {
            fprintf(stderr, "Usage: %s --benchmark mode size blockSize warmups steps keepPercent\n", argv[0]);
            return 1;
        }

        // Read the settings in the same order used by run_profiles.ps1.
        std::string mode = argv[2];
        int n = std::stoi(argv[3]);
        Common::blockSize = std::stoi(argv[4]);
        int warmups = std::stoi(argv[5]);
        int steps = std::stoi(argv[6]);
        int keepPercent = std::stoi(argv[7]);

        // Keep lengths, thread counts, and repetition counts in range.
        if (n < 1 || n > (1 << 26) || Common::blockSize < 1 || Common::blockSize > 1024 ||
            warmups < 0 || steps < 1 || keepPercent < 0 || keepPercent > 100) {
            throw std::runtime_error("Invalid benchmark arguments");
        }

        // Reuse the seed so separate runs get the same input.
        std::mt19937 random(5650);
        std::vector<int> input(n), output(n);
        for (int &value : input) {
            value = static_cast<int>(random() % 100) < keepPercent ? 1 + random() % 4 : 0;
        }

        // Check once before warmup, then run without measuring the warmups.
        verify(mode, input);
        int count;
        for (int i = 0; i < warmups; ++i) {
            run(mode, n, output.data(), input.data(), count);
        }

        // Each implementation reports just its own timed algorithm work.
        double total = 0;
        for (int i = 0; i < steps; ++i) {
            total += run(mode, n, output.data(), input.data(), count);
        }

        // Check the final result too, after all the repeated calls.
        auto expected = expectedResult(input, isCompact(mode));

        if (count != static_cast<int>(expected.size()) ||
            !std::equal(expected.begin(), expected.end(), output.begin())) {
            throw std::runtime_error("Benchmark result is incorrect");
        }

        // The profiling script reads this line as one CSV measurement.
        printf("%s,%d,%d,%d,%d,%.9f\n", mode.c_str(), n, Common::blockSize,
            steps, keepPercent, total / steps);
    } catch (const std::exception &error) {
        fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    return 0;
}
