param(
    [Parameter(Mandatory = $true)][string] $Application,
    [Parameter(Mandatory = $true)][string] $Output,
    [string[]] $ArgumentList = @(),
    [string] $ProcessName = [System.IO.Path]::GetFileNameWithoutExtension($Application),
    [string] $RgpCli = $env:RGP_CLI,
    [int] $Dispatch = 1,
    [int] $DelayMs = 1000,
    [int] $TimeoutSeconds = 60,
    [int] $Attempts = 3,
    [string] $HandshakeFile = "",
    [switch] $NoInstructionTracing
)

$ErrorActionPreference = "Stop"

if (-not $RgpCli) {
    throw "Set RGP_CLI to the RadeonDeveloperPanelCLI.exe path, or pass -RgpCli."
}
if (-not (Test-Path $RgpCli)) {
    throw "RGP CLI not found at $RgpCli."
}

$outputDirectory = Split-Path $Output -Parent
if ($outputDirectory) {
    New-Item -ItemType Directory -Force $outputDirectory | Out-Null
}

function Start-Panel([string[]] $PanelArguments) {
    $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $RgpCli
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardInput = $true
    foreach ($argument in $PanelArguments) {
        $startInfo.ArgumentList.Add($argument)
    }
    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo = $startInfo
    $null = $process.Start()
    return $process
}

function Stop-IfRunning($Process) {
    if ($null -ne $Process -and -not $Process.HasExited) {
        $Process.Kill()
    }
}

$handshake = -not [string]::IsNullOrWhiteSpace($HandshakeFile)

$panelArguments = @(
    "-m", "profiling",
    "-p", $ProcessName,
    "-o", $Output,
    "--verbose"
)
if ($handshake) {
    # Explicit capture request. The application waits until the request has been
    # sent before it dispatches, so the first dispatch is already armed.
    $panelArguments += @("--rgp-capture-mode", "dispatch", "--rgp-render-op-count", "1")
} else {
    $panelArguments += @("-a", "dispatch:$Dispatch:1", "--rgp-auto-capture-delay", "$DelayMs")
}
if (-not $NoInstructionTracing) {
    $panelArguments += "--rgp-instruction-tracing"
}

for ($attempt = 1; $attempt -le $Attempts; ++$attempt) {
    Remove-Item $Output -Force -ErrorAction SilentlyContinue
    if ($handshake) {
        Remove-Item $HandshakeFile -Force -ErrorAction SilentlyContinue
    }

    $panel = Start-Panel $panelArguments
    Start-Sleep -Seconds 3
    if ($panel.HasExited) {
        Write-Warning "RGP panel exited before the probe started on attempt $attempt."
        continue
    }

    $applicationArguments = @($ArgumentList)
    $readyFile = "$HandshakeFile.ready"
    if ($handshake) {
        Remove-Item $readyFile -Force -ErrorAction SilentlyContinue
        $applicationArguments += @("--ready-file", $readyFile, "--wait-file", $HandshakeFile)
    }

    $application = Start-Process -FilePath $Application -ArgumentList $applicationArguments -PassThru -NoNewWindow `
        -RedirectStandardOutput "$Output.app.log" -RedirectStandardError "$Output.app.err"
    $applicationId = if ($application) { $application.Id } else { $null }

    if ($handshake) {
        # The capture request must arrive after the device context exists, otherwise
        # the driver fails the trace with result -2. The probe reports readiness.
        $readyDeadline = (Get-Date).AddSeconds(30)
        while ((Get-Date) -lt $readyDeadline -and -not (Test-Path $readyFile)) {
            if ($applicationId -and -not (Get-Process -Id $applicationId -ErrorAction SilentlyContinue)) { break }
            if ($panel.HasExited) { break }
            Start-Sleep -Milliseconds 100
        }
        if (-not (Test-Path $readyFile)) {
            Stop-Process -Id $applicationId -Force -ErrorAction SilentlyContinue
            Write-Warning "Probe never reported a ready context on attempt $attempt."
            continue
        }
        if ($panel.HasExited) {
            Stop-Process -Id $applicationId -Force -ErrorAction SilentlyContinue
            Write-Warning "RGP panel exited before the capture request on attempt $attempt."
            continue
        }
        try {
            $panel.StandardInput.WriteLine("c")
            $panel.StandardInput.Flush()
        } catch {
            Stop-Process -Id $applicationId -Force -ErrorAction SilentlyContinue
            Write-Warning "Could not send the capture request on attempt $attempt."
            continue
        }
        Start-Sleep -Milliseconds 500
        New-Item -ItemType File -Force $HandshakeFile | Out-Null
    }

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline -and -not (Test-Path $Output)) {
        if ($applicationId -and -not (Get-Process -Id $applicationId -ErrorAction SilentlyContinue)) {
            break
        }
        Start-Sleep -Milliseconds 300
    }

    if ($applicationId -and (Get-Process -Id $applicationId -ErrorAction SilentlyContinue)) {
        Stop-Process -Id $applicationId -Force -ErrorAction SilentlyContinue
    }
    Stop-IfRunning $panel

    if (Test-Path $Output) {
        if ($handshake) {
            Remove-Item $HandshakeFile -Force -ErrorAction SilentlyContinue
            Remove-Item $readyFile -Force -ErrorAction SilentlyContinue
        }
        Get-Item $Output | Select-Object FullName, Length
        exit 0
    }

    Write-Warning "RGP produced no capture on attempt $attempt."
}

throw "No capture produced after $Attempts attempts. Check the panel output and $Output.app.log."
