param(
    [string]$Executable = ".\build\bin\Release\cis5650_stream_compaction_test.exe",
    [string]$Nsys = "C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.1.3\target-windows-x64\nsys.exe",
    [string]$Ncu = "C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.2.1\ncu.bat"
)
$ErrorActionPreference = "Stop"
$results = Join-Path $PSScriptRoot "results"

# Use the block sizes chosen by the ordinary timing runs.
$sizes = Get-Content "$results/block_sizes.json" | ConvertFrom-Json

# Systems shows when kernels and CUDA API calls run relative to each other.
foreach ($mode in @("naive", "efficient", "thrust")) {
    $block = if ($mode -eq "thrust") { 128 } else { $sizes.$mode }
    & $Nsys profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite=true `
        --output="$results/nsight_systems_$mode" $Executable --benchmark $mode 1048576 $block 2 3 50
    if ($LASTEXITCODE -ne 0) { throw "Nsight Systems failed for $mode" }

    # Export summaries and the SQLite data used by the timeline plot.
    & $Nsys stats --report cuda_gpu_kern_sum,cuda_api_sum --format csv `
        "$results/nsight_systems_$mode.nsys-rep" |
        Set-Content -Encoding utf8 "$results/nsight_systems_$mode.txt"
    if ($LASTEXITCODE -ne 0) { throw "Nsight stats failed for $mode" }
}

# Compute replays one matching kernel to collect hardware counters.
# These runs are separate from the benchmark averages.
foreach ($mode in @("naive", "efficient")) {
    $kernel = if ($mode -eq "naive") { "kernAddPrevious" } else { "kernUpsweep" }
    & $Ncu --set detailed --kernel-name-base function --kernel-name "regex:.*$kernel.*" `
        --launch-count 1 --export "$results/nsight_compute_$mode" --force-overwrite `
        $Executable --benchmark $mode 1048576 $sizes.$mode 0 1 50
    if ($LASTEXITCODE -ne 0) { throw "Nsight Compute failed for $mode" }

    # Save the counters in a format the plotting script can read.
    & $Ncu --import "$results/nsight_compute_$mode.ncu-rep" --page raw --csv |
        Set-Content -Encoding utf8 "$results/nsight_compute_$mode.csv"
}
