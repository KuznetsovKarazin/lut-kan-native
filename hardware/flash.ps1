#Requires -Version 5.1
<#
.SYNOPSIS
    Flash lut-kan-native benchmark to Arduino Mega 2560 or ESP32-C3.

.PARAMETER Device
    mega | esp32c3 | auto | both

.PARAMETER Port
    COM port override, e.g. -Port COM4

.PARAMETER Monitor
    Open serial monitor after flashing (115200 baud). Press Ctrl+C to exit.

.PARAMETER SkipInstall
    Skip arduino-cli install/core update (faster if already set up).

.PARAMETER ShowOutput
    Show full arduino-cli compiler output.

.EXAMPLE
    .\hardware\flash.ps1 -Monitor
    .\hardware\flash.ps1 -Device mega -Port COM3 -Monitor
    .\hardware\flash.ps1 -Device both -Monitor
    .\hardware\flash.ps1 -SkipInstall -Monitor
#>
param(
    [ValidateSet("mega","esp32c3","auto","both")]
    [string]$Device = "auto",
    [ValidateSet("bench","ntc")]
    [string]$Sketch = "bench",  # bench = accuracy benchmark, ntc = thermistor use case
    [string]$Port = "",
    [switch]$Monitor,
    [switch]$SkipInstall,
    [switch]$ShowOutput
)

Set-StrictMode -Version 1
$ErrorActionPreference = "Stop"

function Write-Step { param($m) Write-Host "  >> $m" -ForegroundColor Cyan   }
function Write-OK   { param($m) Write-Host "  OK $m" -ForegroundColor Green  }
function Write-Warn { param($m) Write-Host "  !! $m" -ForegroundColor Yellow }
function Write-Fail { param($m) Write-Host " ERR $m" -ForegroundColor Red    }

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
if ($Sketch -eq "ntc") {
    $SketchDir  = Join-Path $ScriptDir "lut_kan_ntc_calib"
    $SketchFile = Join-Path $SketchDir "lut_kan_ntc_calib.ino"
} else {
    $SketchDir  = Join-Path $ScriptDir "lut_kan_hw_bench"
    $SketchFile = Join-Path $SketchDir "lut_kan_hw_bench.ino"
}

$CliDir = Join-Path $env:LOCALAPPDATA "arduino-cli"
$CliExe = Join-Path $CliDir "arduino-cli.exe"

$FQBN_MEGA    = "arduino:avr:mega:cpu=atmega2560"
$FQBN_ESP32C3 = "esp32:esp32:esp32c3:CDCOnBoot=cdc,CPUFreq=160,FlashMode=dio,FlashFreq=80,FlashSize=4M,PartitionScheme=default,DebugLevel=none,EraseFlash=none"
$ESP32_URL    = "https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json"

# ---------------------------------------------------------------------------
# arduino-cli wrapper
# ---------------------------------------------------------------------------
function Run-CLI {
    param([string[]]$A)
    if ($ShowOutput) {
        & $CliExe @A
    } else {
        & $CliExe @A 2>&1 | Out-Null
    }
    return $LASTEXITCODE
}

function Get-CLIOut {
    param([string[]]$A)
    return (& $CliExe @A 2>&1)
}

# ---------------------------------------------------------------------------
# Step 1 - Sketch structure
# ---------------------------------------------------------------------------
function Ensure-SketchStructure {
    Write-Step "Checking sketch..."
    if (-not (Test-Path $SketchDir)) {
        New-Item -ItemType Directory -Path $SketchDir | Out-Null
    }
    $candidates = @(
        (Join-Path $ScriptDir "lut_kan_hw_bench.ino"),
        $SketchFile
    )
    $found = $null
    foreach ($c in $candidates) {
        if (Test-Path $c) { $found = $c; break }
    }
    if ($null -eq $found) {
        Write-Fail "lut_kan_hw_bench.ino not found."
        Write-Fail "Expected at: $SketchFile"
        exit 1
    }
    if ($found -ne $SketchFile) {
        Copy-Item $found $SketchFile -Force
    }
    Write-OK "Sketch: $SketchFile"
}

# ---------------------------------------------------------------------------
# Step 2 - Install arduino-cli
# ---------------------------------------------------------------------------
function Ensure-CLI {
    if (-not (Test-Path $CliExe)) {
        $onPath = Get-Command arduino-cli -ErrorAction SilentlyContinue
        if ($onPath) {
            $script:CliExe = $onPath.Source
        }
    }

    if (Test-Path $CliExe) {
        $v = (& $CliExe version 2>&1) -join ""
        Write-OK "arduino-cli: $v"
        return
    }

    Write-Step "arduino-cli not found, installing..."

    $wg = Get-Command winget -ErrorAction SilentlyContinue
    if ($wg) {
        Write-Step "Trying winget..."
        winget install --id ArduinoSA.CLI --silent --accept-package-agreements --accept-source-agreements 2>&1 | Out-Null
        $onPath = Get-Command arduino-cli -ErrorAction SilentlyContinue
        if ($onPath) {
            $script:CliExe = $onPath.Source
            Write-OK "Installed via winget"
            return
        }
    }

    Write-Step "Downloading from GitHub releases..."
    New-Item -ItemType Directory -Path $CliDir -Force | Out-Null

    $rel   = Invoke-RestMethod "https://api.github.com/repos/arduino/arduino-cli/releases/latest" -UseBasicParsing
    $asset = $rel.assets | Where-Object { $_.name -like "*Windows_64bit.zip" } | Select-Object -First 1
    if ($null -eq $asset) {
        Write-Fail "Cannot find Windows zip. Install manually:"
        Write-Fail "https://arduino.github.io/arduino-cli/latest/installation/"
        exit 1
    }

    $zip = Join-Path $env:TEMP "arduino-cli.zip"
    Write-Step "Downloading $($asset.browser_download_url)..."
    Invoke-WebRequest $asset.browser_download_url -OutFile $zip -UseBasicParsing
    Expand-Archive $zip $CliDir -Force
    Remove-Item $zip

    if (-not (Test-Path $CliExe)) {
        Write-Fail "Extraction failed. Check $CliDir"
        exit 1
    }
    Write-OK "arduino-cli installed: $CliExe"
}

# ---------------------------------------------------------------------------
# Step 3 - Config + cores
# ---------------------------------------------------------------------------
function Ensure-Config {
    Write-Step "Checking config..."
    $urls = (Get-CLIOut @("config","get","board_manager.additional_urls")) -join ""
    if ($urls -notlike "*espressif*") {
        Write-Step "Adding ESP32 board manager URL..."
        & $CliExe config add board_manager.additional_urls $ESP32_URL 2>&1 | Out-Null
    }
    Write-OK "Config ready"
}

function Ensure-Cores {
    Write-Step "Updating board index..."
    & $CliExe core update-index 2>&1 | Out-Null

    $installed = (Get-CLIOut @("core","list")) -join ""

    if ($installed -notlike "*arduino:avr*") {
        Write-Step "Installing arduino:avr..."
        & $CliExe core install arduino:avr 2>&1 | Out-Null
        Write-OK "arduino:avr installed"
    } else {
        Write-OK "arduino:avr ready"
    }

    if ($installed -notlike "*esp32:esp32*") {
        Write-Step "Installing esp32:esp32 (~300 MB, one-time, please wait)..."
        & $CliExe core install esp32:esp32 2>&1 | Out-Null
        Write-OK "esp32:esp32 installed"
    } else {
        Write-OK "esp32:esp32 ready"
    }
}

# ---------------------------------------------------------------------------
# Step 4 - Port detection
# ---------------------------------------------------------------------------
function Get-Boards {
    $raw = (& $CliExe board list --json 2>&1) -join ""
    try {
        $parsed = $raw | ConvertFrom-Json
        # Force array - single item comes back as PSCustomObject, not array
        return @($parsed.detected_ports)
    } catch {
        return @()
    }
}

function Find-Port {
    param([string]$DevType)
    if ($Port -ne "") { return $Port }

    Write-Step "Detecting port for $DevType..."
    $boards = Get-Boards
    if ($null -ne $boards) {
        foreach ($b in $boards) {
            $name = ""
            if ($b.matching_boards -and $b.matching_boards.Count -gt 0) {
                $name = $b.matching_boards[0].name
            }
            if ($DevType -eq "mega"    -and $name -like "*Mega*")       { return $b.port.address }
            if ($DevType -eq "esp32c3" -and ($name -like "*ESP32*C3*" -or $name -like "*ESP32C3*" -or $name -like "*ESP32-C3*" -or $name -like "*esp32c3*")) { return $b.port.address }
        }
    }
    return $null
}

function Prompt-Port {
    param([string]$DevType)
    Write-Host ""
    Write-Warn "Cannot auto-detect port for $DevType. Showing visible ports:"
    $boards = Get-Boards
    if ($null -ne $boards -and $boards.Count -gt 0) {
        foreach ($b in $boards) {
            $name = ""
            if ($b.matching_boards -and $b.matching_boards.Count -gt 0) {
                $name = $b.matching_boards[0].name
            }
            Write-Host "    $($b.port.address)  $name" -ForegroundColor Gray
        }
    } else {
        $wmi = Get-WmiObject Win32_PnPEntity -Filter "Name LIKE '%(COM%'" -ErrorAction SilentlyContinue
        if ($wmi) {
            foreach ($p in $wmi) { Write-Host "    $($p.Name)" -ForegroundColor Gray }
        }
    }
    Write-Host ""
    Write-Host "  Enter COM port (e.g. COM3):" -ForegroundColor Yellow
    $r = Read-Host "  Port"
    return $r.Trim()
}

# ---------------------------------------------------------------------------
# Step 5 - Compile
# ---------------------------------------------------------------------------
function Invoke-Compile {
    param([string]$Fqbn, [string]$Label)
    Write-Step "Compiling for $Label..."
    $bd = Join-Path $env:TEMP "lut_kan_build"
    New-Item -ItemType Directory -Path $bd -Force | Out-Null

    $out = & $CliExe compile --fqbn $Fqbn --build-path $bd $SketchDir 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Fail "Compile failed:"
        Write-Host ($out -join "`n") -ForegroundColor Red
        return $null
    }
    Write-OK "Compiled OK"
    return $bd
}

# ---------------------------------------------------------------------------
# Step 6 - Upload
# ---------------------------------------------------------------------------
function Invoke-Upload {
    param([string]$Fqbn, [string]$PortAddr, [string]$BuildDir, [string]$Label)
    Write-Step "Uploading to $Label on $PortAddr..."
    $out = & $CliExe upload --fqbn $Fqbn --port $PortAddr --input-dir $BuildDir $SketchDir 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Fail "Upload failed:"
        Write-Host ($out -join "`n") -ForegroundColor Yellow
        if ($Label -like "*ESP32*") {
            Write-Host ""
            Write-Warn "ESP32-C3 boot mode hint:"
            Write-Host "  1. Hold BOOT button" -ForegroundColor Yellow
            Write-Host "  2. Press and release RESET" -ForegroundColor Yellow
            Write-Host "  3. Release BOOT" -ForegroundColor Yellow
            Write-Host "  4. Re-run this script" -ForegroundColor Yellow
        }
        return $false
    }
    Write-OK "Upload complete"
    return $true
}

# ---------------------------------------------------------------------------
# Step 7 - Serial monitor (built-in .NET, no extra tools needed)
# ---------------------------------------------------------------------------
function Open-Monitor {
    param([string]$PortAddr)
    Write-Host ""
    Write-Host "  -- Serial Monitor on $PortAddr at 115200 baud --" -ForegroundColor DarkCyan
    Write-Host "  Press Ctrl+C to exit" -ForegroundColor Gray
    Write-Host ""

    $sp = $null
    try {
        $sp = New-Object System.IO.Ports.SerialPort($PortAddr, 115200, "None", 8, "One")
        $sp.ReadTimeout = 500
        $sp.NewLine     = "`n"
        $sp.Open()

        $sp.DtrEnable = $false; Start-Sleep -Milliseconds 100
        $sp.DtrEnable = $true;  Start-Sleep -Milliseconds 200

        $t0      = Get-Date
        $gotData = $false

        while ($true) {
            try {
                $line    = $sp.ReadLine().TrimEnd()
                $gotData = $true
                Write-Host "  $line"
            } catch [System.TimeoutException] {
                if ((-not $gotData) -and ((Get-Date) - $t0).TotalSeconds -gt 8) {
                    Write-Warn "No output after 8s -- press Reset on the board"
                    $t0 = Get-Date
                }
            }
        }
    } catch {
        $msg = "$_"
        if ($msg -notlike "*pipeline*" -and $msg -notlike "*stopped*") {
            Write-Warn "Monitor closed: $msg"
        }
    } finally {
        if ($null -ne $sp -and $sp.IsOpen) { $sp.Close() }
        Write-Host ""
        Write-Host "  Monitor closed." -ForegroundColor Gray
    }
}

# ---------------------------------------------------------------------------
# Step 8 - Flash one device
# ---------------------------------------------------------------------------
function Flash-Device {
    param([string]$DevType)

    $fqbn  = if ($DevType -eq "mega") { $FQBN_MEGA } else { $FQBN_ESP32C3 }
    $label = if ($DevType -eq "mega") { "Arduino Mega 2560" } else { "ESP32-C3 SuperMini" }

    Write-Host ""
    Write-Host "  ========================================" -ForegroundColor DarkCyan
    Write-Host "  Target: $label" -ForegroundColor DarkCyan
    Write-Host "  ========================================" -ForegroundColor DarkCyan

    $bd = Invoke-Compile -Fqbn $fqbn -Label $label
    if ($null -eq $bd) { return $false }

    $p = Find-Port -DevType $DevType
    if ([string]::IsNullOrEmpty($p)) {
        $p = Prompt-Port -DevType $DevType
    }
    if ([string]::IsNullOrEmpty($p)) {
        Write-Warn "No port -- skipping $label"
        return $false
    }
    Write-OK "Port: $p"

    $ok = Invoke-Upload -Fqbn $fqbn -PortAddr $p -BuildDir $bd -Label $label
    if (-not $ok) { return $false }

    if ($Monitor) {
        Start-Sleep -Seconds 2
        Open-Monitor -PortAddr $p
    } else {
        Write-Host ""
        Write-OK "Done. Open serial monitor at 115200 baud on $p"
        Write-Host "  Quick monitor: arduino-cli monitor --port $p --config baudrate=115200" -ForegroundColor Gray
    }
    return $true
}

# ---------------------------------------------------------------------------
# Auto-detect
# ---------------------------------------------------------------------------
function Get-AutoDevice {
    $boards = Get-Boards
    if ($null -eq $boards) { return "unknown" }
    foreach ($b in $boards) {
        $name = ""
        if ($b.matching_boards -and $b.matching_boards.Count -gt 0) {
            $name = $b.matching_boards[0].name
        }
        if ($name -like "*Mega*") { return "mega" }
        if ($name -like "*ESP32*C3*" -or $name -like "*ESP32C3*" -or $name -like "*ESP32-C3*" -or $name -like "*esp32c3*") { return "esp32c3" }
    }
    return "unknown"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "  lut-kan-native hardware flash" -ForegroundColor Cyan
Write-Host "  Sketch : $SketchFile" -ForegroundColor Gray
Write-Host "  Device : $Device" -ForegroundColor Gray
Write-Host ""

Ensure-SketchStructure

if (-not $SkipInstall) {
    Ensure-CLI
    Ensure-Config
    Ensure-Cores
} else {
    Write-Warn "SkipInstall set -- assuming arduino-cli and cores are ready"
    if (-not (Test-Path $CliExe)) {
        $onPath = Get-Command arduino-cli -ErrorAction SilentlyContinue
        if ($onPath) {
            $script:CliExe = $onPath.Source
        } else {
            Write-Fail "arduino-cli not found. Remove -SkipInstall to auto-install."
            exit 1
        }
    }
}

$targets = @()
switch ($Device) {
    "mega"    { $targets = @("mega") }
    "esp32c3" { $targets = @("esp32c3") }
    "both"    { $targets = @("mega","esp32c3") }
    "auto"    {
        Write-Step "Auto-detecting connected device..."
        $det = Get-AutoDevice
        if ($det -eq "unknown") {
            $boards = Get-Boards
            if ($boards.Count -gt 0) {
                Write-Host ""
                Write-Host "  Found ports (could not match to known board):" -ForegroundColor Gray
                foreach ($b in $boards) {
                    $name = ""
                    if ($b.matching_boards -and $b.matching_boards.Count -gt 0) {
                        $name = $b.matching_boards[0].name
                    }
                    Write-Host "    $($b.port.address)  $name" -ForegroundColor Gray
                }
                Write-Host ""
                Write-Host "  Which device is connected? (mega / esp32c3):" -ForegroundColor Yellow
                $ans = (Read-Host "  Device").Trim().ToLower()
                if ($ans -eq "mega" -or $ans -eq "esp32c3") {
                    $det = $ans
                } else {
                    Write-Fail "Unknown answer. Use -Device mega or -Device esp32c3 explicitly."
                    exit 1
                }
            } else {
                Write-Warn "No device detected at all. Plug in and retry, or use -Device mega|esp32c3"
                exit 1
            }
        }
        Write-OK "Detected: $det"
        $targets = @($det)
    }
}

$allOk = $true
for ($i = 0; $i -lt $targets.Count; $i++) {
    if ($i -gt 0) {
        Write-Host ""
        Write-Host "  Connect next device ($($targets[$i])) and press Enter..." -ForegroundColor Yellow
        Read-Host | Out-Null
    }
    $ok = Flash-Device -DevType $targets[$i]
    if (-not $ok) { $allOk = $false }
}

Write-Host ""
if ($allOk) {
    Write-Host "  All done." -ForegroundColor Green
} else {
    Write-Host "  Some steps failed -- check messages above." -ForegroundColor Yellow
    exit 1
}
