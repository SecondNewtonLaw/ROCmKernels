param(
    [Parameter(Mandatory = $true)][string] $Source,
    [Parameter(Mandatory = $true)][string] $Output,
    [string] $RocmPath = $env:ROCM_PATH,
    [string] $Architecture = "gfx1031",
    [string[]] $IncludePath = @()
)

$ErrorActionPreference = "Stop"

if (-not $RocmPath) {
    if ($env:VIRTUAL_ENV) {
        $candidate = Join-Path $env:VIRTUAL_ENV "Lib/site-packages/_rocm_sdk_core"
        if (Test-Path (Join-Path $candidate "include/hip/hip_runtime.h")) {
            $RocmPath = $candidate
        }
    }
}
if (-not $RocmPath) {
    $hipconfig = Get-Command hipconfig -ErrorAction SilentlyContinue
    if ($hipconfig) {
        $candidate = (& $hipconfig.Source --path 2>$null | Select-Object -First 1)
        if ($LASTEXITCODE -eq 0 -and $candidate -and (Test-Path $candidate.Trim())) {
            $RocmPath = $candidate.Trim()
        }
    }
}
if (-not $RocmPath -or -not (Test-Path $RocmPath)) {
    throw "Set ROCM_PATH, pass -RocmPath, or put a working hipconfig on PATH."
}

$toolDirectories = @(
    (Join-Path $RocmPath "lib/llvm/bin"),
    (Join-Path $RocmPath "bin")
)
$toolDirectory = $toolDirectories | Where-Object { Test-Path (Join-Path $_ "clang++.exe") } | Select-Object -First 1
if (-not $toolDirectory) {
    throw "No clang++.exe found under $RocmPath."
}
$clang = Join-Path $toolDirectory "clang++.exe"

$bitcodeCandidates = @(
    (Join-Path $RocmPath "lib/llvm/amdgcn/bitcode"),
    (Join-Path $RocmPath "amdgcn/bitcode")
)
$bitcode = $bitcodeCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $bitcode) {
    throw "No device bitcode found under $RocmPath."
}

$repositoryRoot = Split-Path $PSScriptRoot -Parent
$outputDirectory = Split-Path $Output -Parent
if ($outputDirectory) {
    New-Item -ItemType Directory -Force $outputDirectory | Out-Null
}

$arguments = @(
    "-x", "hip",
    "--rocm-device-lib-path=$bitcode",
    "--offload-arch=$Architecture",
    "-S", "--cuda-device-only",
    "-O3",
    "-std=c++23",
    "-ffp-contract=off",
    "-D__HIP_PLATFORM_AMD__",
    "-I", (Join-Path $repositoryRoot "ROCmKernels.Kernels/include"),
    "-I", (Join-Path $repositoryRoot "ROCmKernels.Common/include"),
    "-isystem", (Join-Path $RocmPath "include")
)
foreach ($path in $IncludePath) {
    $arguments += @("-I", $path)
}
$arguments += @($Source, "-o", $Output)

& $clang @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Disassembly failed."
}

Write-Output $Output
