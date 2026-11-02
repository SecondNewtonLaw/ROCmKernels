param(
    [string] $AssemblyDirectory = (Join-Path $PSScriptRoot "..\build\local-release\assembly")
)

$files = Get-ChildItem -Path $AssemblyDirectory -Filter *.s -ErrorAction SilentlyContinue | Sort-Object Name
if (-not $files) {
    Write-Error "No assembly files found in $AssemblyDirectory. Build the ROCmKernels.Assembly target first."
    exit 1
}

function Get-InstructionCategory([string] $mnemonic) {
    if ($mnemonic -match '^(s_branch|s_cbranch|s_setpc|s_swappc|s_endpgm)') { return 'Branch' }
    if ($mnemonic -match '^(global|flat|scratch|buffer|tbuffer|s_buffer)_') { return 'VMEM' }
    if ($mnemonic -match '^ds_') { return 'LDS' }
    if ($mnemonic -match '^v_') { return 'VALU' }
    if ($mnemonic -match '^s_') { return 'SALU' }
    return 'Other'
}

$functionPattern = '^([A-Za-z_][A-Za-z0-9_.$]*):\s*(;.*)?$'
$instructionPattern = '^\s+([a-z][a-z0-9_]+)\b'

foreach ($file in $files) {
    $functions = [ordered]@{}
    $current = $null

    foreach ($line in Get-Content $file.FullName) {
        if ($line -match $functionPattern) {
            $name = $Matches[1]
            if ($name -like '.L*') { continue }
            $current = $name
            if (-not $functions.Contains($current)) {
                $functions[$current] = @{ Total = 0; VALU = 0; SALU = 0; VMEM = 0; LDS = 0; Branch = 0; Other = 0 }
            }
            continue
        }

        if ($null -eq $current) { continue }
        if ($line -notmatch $instructionPattern) { continue }

        $mnemonic = $Matches[1]
        $category = Get-InstructionCategory $mnemonic
        $functions[$current].Total++
        $functions[$current][$category]++
    }

    $fileTotal = 0
    foreach ($name in $functions.Keys) { $fileTotal += $functions[$name].Total }
    if ($fileTotal -eq 0) { continue }

    Write-Output ""
    Write-Output $file.Name
    foreach ($name in $functions.Keys) {
        $stats = $functions[$name]
        if ($stats.Total -eq 0) { continue }
        Write-Output ("  {0,-44} {1,6}  VALU {2,5}  SALU {3,5}  VMEM {4,5}  LDS {5,4}  Branch {6,4}  Other {7,5}" -f `
            $name, $stats.Total, $stats.VALU, $stats.SALU, $stats.VMEM, $stats.LDS, $stats.Branch, $stats.Other)
    }
}
