param(
    [string]$Executable = ".\build\bin\Release\cis5650_stream_compaction_test.exe",
    [int]$Trials = 3
)
$ErrorActionPreference = "Stop"
$results = Join-Path $PSScriptRoot "results"
New-Item -ItemType Directory -Force $results | Out-Null
# Collect one row per run, so we can calculate variation later.
$rows = [System.Collections.Generic.List[object]]::new()

function Measure-Scan {
    param([string]$Experiment, [string]$Mode, [int]$Size, [int]$BlockSize, [int]$Keep, [int]$Trial)

    # Each process warms up 20 times, then measures 100 operations.
    $line = & $Executable --benchmark $Mode $Size $BlockSize 20 100 $Keep
    if ($LASTEXITCODE -ne 0) { throw "Benchmark failed: $Mode $Size" }

    # Read the average time from the CSV line printed by the program.
    $parts = ($line | Select-Object -Last 1) -split ','
    $row = [pscustomobject]@{
        experiment = $Experiment
        mode = $Mode
        size = $Size
        block_size = $BlockSize
        keep_percent = $Keep
        trial = $Trial
        measured_steps = 100
        ms = [double]$parts[5]
    }
    $rows.Add($row)
    Write-Host "$Experiment $Mode size=$Size block=$BlockSize trial=$Trial ms=$($row.ms)"
}
# Use the largest input to choose a block size before comparing the implementations.
foreach ($trial in 1..$Trials) {
    foreach ($block in @(32, 64, 128, 256, 512, 1024)) {
        foreach ($mode in @("naive", "efficient", "gpu-compact")) {
            Measure-Scan "block_size" $mode 4194304 $block 50 $trial
        }
    }
}

# Choose the block size with the lowest average across all trials.
$chosen = @{}
foreach ($mode in @("naive", "efficient", "gpu-compact")) {
    $best = $rows | Where-Object mode -eq $mode | Group-Object block_size |
        Sort-Object { ($_.Group | Measure-Object ms -Average).Average } | Select-Object -First 1
    $chosen[$mode] = [int]$best.Name
}

# Save these choices so Nsight uses the same block sizes.
$chosen | ConvertTo-Json | Set-Content "$results/block_sizes.json"
foreach ($trial in 1..$Trials) {
    # Compare lengths using the block sizes selected above.
    foreach ($size in @(256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304)) {
        foreach ($mode in @("cpu", "naive", "efficient", "thrust", "cpu-compact", "cpu-scan-compact", "gpu-compact")) {
            $block = if ($chosen.ContainsKey($mode)) { $chosen[$mode] } else { 128 }
            Measure-Scan "array_size" $mode $size $block 50 $trial
        }
    }

    # Hold length fixed and change how many values are likely to survive.
    foreach ($keep in @(0, 25, 50, 75, 100)) {
        foreach ($mode in @("cpu-compact", "cpu-scan-compact", "gpu-compact")) {
            $block = if ($chosen.ContainsKey($mode)) { $chosen[$mode] } else { 128 }
            Measure-Scan "keep_percent" $mode 1048576 $block $keep $trial
        }
    }

    # Adding one element above a power of two doubles the padded tree.
    foreach ($size in @(1048575, 1048576, 1048577)) {
        foreach ($mode in @("naive", "efficient", "thrust")) {
            $block = if ($chosen.ContainsKey($mode)) { $chosen[$mode] } else { 128 }
            Measure-Scan "padding" $mode $size $block 50 $trial
        }
    }
}

# Keep the raw runs as well as the averages used in the graphs.
$rows | Export-Csv -NoTypeInformation "$results/timings.csv"
