param(
    [Parameter(Mandatory = $true)][string] $Source,
    [Parameter(Mandatory = $true)][string] $Output,
    [string] $RocmPath = $env:ROCM_PATH,
    [string] $Architecture = "gfx1031",
    [string[]] $IncludePath = @(),
    [string[]] $Define = @()
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
$objcopy = Join-Path $toolDirectory "llvm-objcopy.exe"
$bundler = Join-Path $toolDirectory "clang-offload-bundler.exe"

$bitcodeCandidates = @(
    (Join-Path $RocmPath "lib/llvm/amdgcn/bitcode"),
    (Join-Path $RocmPath "amdgcn/bitcode")
)
$bitcode = $bitcodeCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $bitcode) {
    throw "No gfx1031 device bitcode found under $RocmPath."
}

$repositoryRoot = Split-Path $PSScriptRoot -Parent
$outputDirectory = Split-Path $Output -Parent
if ($outputDirectory) {
    New-Item -ItemType Directory -Force $outputDirectory | Out-Null
}

$objectPath = "$Output.o"
$fatPath = "$Output.fatbin"

$compileArguments = @(
    "-x", "hip",
    "--rocm-device-lib-path=$bitcode",
    "--offload-arch=$Architecture",
    "-O3",
    "-std=c++23",
    "-ffp-contract=off",
    "-isystem", (Join-Path $RocmPath "include"),
    "-I", (Join-Path $repositoryRoot "ROCmKernels.Kernels/include"),
    "-I", (Join-Path $repositoryRoot "ROCmKernels.Common/include")
)
foreach ($path in $IncludePath) {
    $compileArguments += @("-I", $path)
}
foreach ($definition in $Define) {
    $compileArguments += "-D$definition"
}
$compileArguments += @("-c", $Source, "-o", $objectPath)

& $clang @compileArguments
if ($LASTEXITCODE -ne 0) {
    throw "HIP device compilation failed."
}

& $objcopy "--dump-section" ".hip_fatbin=$fatPath" $objectPath
if ($LASTEXITCODE -ne 0) {
    throw "Could not extract .hip_fatbin from $objectPath."
}

& $bundler "-unbundle" "-type" "o" "-targets" "hip-amdgcn-amd-amdhsa--$Architecture" "-input" $fatPath "-output" $Output
if ($LASTEXITCODE -ne 0) {
    throw "Could not unbundle the code object."
}

Remove-Item $objectPath, $fatPath -Force -ErrorAction SilentlyContinue
Write-Output $Output
