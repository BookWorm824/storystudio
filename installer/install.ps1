# StoryStudio installer (per-user, no admin rights).
# Installs into %USERPROFILE%\StoryStudio (not AppData: packaged apps hide AppData writes
# from shortcuts) with its own Python environment, then adds Start Menu and Desktop shortcuts.
# Safe to run again: it updates in place and keeps settings, shot lists and downloads.
# After installing, StoryStudio updates itself from GitHub (see ss_update.py).
param(
    [switch]$CheckOnly,          # testing: copy the files and find Python, then stop
    [string]$InstallTo = ""      # testing: install somewhere else
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$Home2   = if ($InstallTo) { $InstallTo } else { Join-Path $env:USERPROFILE "StoryStudio" }
$AppDir  = Join-Path $Home2 "app"
$Venv    = Join-Path $Home2 "venv"
$BinDir  = Join-Path $Home2 "bin"
# the app files: the "app" folder next to this script (setup zip), or the folder above it
$Src = Join-Path $PSScriptRoot "app"
if (-not (Test-Path (Join-Path $Src "version.json"))) { $Src = Split-Path -Parent $PSScriptRoot }

New-Item -ItemType Directory -Force $Home2, $AppDir, $BinDir | Out-Null
try { Start-Transcript -Path (Join-Path $Home2 "install_log.txt") -Force | Out-Null } catch {}

function Say($m) { Write-Host ""; Write-Host "== $m" -ForegroundColor Cyan }
function Fail($m) {
    Write-Host ""
    Write-Host "StoryStudio couldn't finish installing: $m" -ForegroundColor Red
    Write-Host "Check the internet connection and run 'Install StoryStudio' again. It picks up where it stopped."
    Write-Host "(Details are saved in $Home2\install_log.txt)"
    try { Stop-Transcript | Out-Null } catch {}
    exit 1
}
function Check($what) { if ($LASTEXITCODE -ne 0) { Fail $what } }

# ------------------------------------------------------------------ the app files
Say "Copying StoryStudio"
$verFile = Join-Path $Src "version.json"
if (-not (Test-Path $verFile)) { Fail "the StoryStudio files are missing from this folder (no version.json)." }
$ver = Get-Content $verFile -Raw -Encoding UTF8 | ConvertFrom-Json
$files = @($ver.files) + @("version.json", "storysound.ico") | Select-Object -Unique
$missing = @()
foreach ($f in $files) {
    $p = Join-Path $Src $f
    if (Test-Path $p) { Copy-Item $p $AppDir -Force } else { $missing += $f }
}
$missing = $missing | Where-Object { $_ -ne "storysound.ico" }
if ($missing.Count -gt 0) { Fail ("some StoryStudio files are missing: " + ($missing -join ", ")) }
Write-Host "Version $($ver.version), $($files.Count) files."

# ------------------------------------------------------------------ Python 3.12
Say "Finding Python 3.12"
function Find-Python {
    $cands = @(
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"),
        (Join-Path $env:ProgramFiles "Python312\python.exe")
    )
    foreach ($c in $cands) { if (Test-Path $c) { return $c } }
    foreach ($c in @("py", "python")) {
        try {
            $out = if ($c -eq "py") { & py -3.12 -c "import sys; print(sys.executable)" 2>$null }
                   else { & python -c "import sys; print(sys.version_info[:2] == (3, 12) and sys.executable or '')" 2>$null }
            if ($LASTEXITCODE -eq 0 -and $out -and (Test-Path $out.Trim())) { return $out.Trim() }
        } catch {}
    }
    return $null
}
$py = Find-Python
if (-not $py) {
    Say "Installing Python 3.12 (one time)"
    $ok = $false
    try {
        winget install -e --id Python.Python.3.12 --scope user --silent --accept-package-agreements --accept-source-agreements | Out-Host
        $py = Find-Python; if ($py) { $ok = $true }
    } catch {}
    if (-not $ok) {
        # no winget (or it failed): use the official installer from python.org
        $pyexe = Join-Path $env:TEMP "python-3.12-installer.exe"
        try {
            Invoke-WebRequest "https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe" -OutFile $pyexe
        } catch { Fail "Python couldn't be downloaded." }
        Start-Process $pyexe -ArgumentList "/quiet", "InstallAllUsers=0", "Include_launcher=1", "Include_tcltk=1", "PrependPath=0" -Wait
        Remove-Item $pyexe -Force -ErrorAction SilentlyContinue
        $py = Find-Python
    }
    if (-not $py) { Fail "Python 3.12 couldn't be installed." }
}
Write-Host "Using $py"
if ($CheckOnly) {
    Write-Host "CHECK ONLY: files copied to $AppDir and Python found; stopping here."
    try { Stop-Transcript | Out-Null } catch {}
    exit 0
}

# ------------------------------------------------------------------ StoryStudio's own Python
$vpy = Join-Path $Venv "Scripts\python.exe"
if (-not (Test-Path $vpy)) {
    Say "Making StoryStudio's own Python environment"
    & $py -m venv $Venv; Check "the Python environment couldn't be made."
}
& $vpy -m pip install --upgrade pip --quiet; Check "pip couldn't be updated."

Say "Installing the basics"
& $vpy -m pip install --quiet numpy pillow soundfile scipy requests sherpa-onnx transformers diffusers accelerate demucs pypdf
Check "the basic Python packages couldn't be installed."

Say "Installing the sound-making AI (large download, one time; this can take a while)"
$hasNvidia = $false
try { if ((Get-CimInstance Win32_VideoController).Name -match "NVIDIA") { $hasNvidia = $true } } catch {}
$haveTorch = $false
try { & $vpy -c "import torch, torchaudio" 2>$null; if ($LASTEXITCODE -eq 0) { $haveTorch = $true } } catch {}
if (-not $haveTorch) {
    if ($hasNvidia) {
        & $vpy -m pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    } else {
        & $vpy -m pip install --quiet torch torchvision torchaudio
    }
    Check "PyTorch (the AI engine) couldn't be installed."
}
& $vpy -m pip install --quiet "https://github.com/hkchengrex/MMAudio/archive/refs/heads/main.zip"
Check "MMAudio (the sound-making AI) couldn't be installed."

# ------------------------------------------------------------------ ffmpeg
Say "Getting ffmpeg (reads and writes video)"
if (-not (Test-Path (Join-Path $BinDir "ffmpeg.exe"))) {
    $found = $false
    # reuse CleanView's / CleanListen's copy if this PC has one (saves a download)
    foreach ($d in @("$env:LOCALAPPDATA\CleanView\app\bin", "$env:LOCALAPPDATA\CleanListen\app\bin")) {
        if ((Test-Path "$d\ffmpeg.exe") -and (Test-Path "$d\ffprobe.exe")) {
            Copy-Item "$d\ffmpeg.exe", "$d\ffprobe.exe" $BinDir -Force
            $found = $true; break
        }
    }
    if (-not $found) {
        $zip = Join-Path $env:TEMP "ffmpeg-storystudio.zip"
        $tmp = Join-Path $env:TEMP "ffmpeg-storystudio"
        try {
            Invoke-WebRequest "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip" -OutFile $zip
            Expand-Archive $zip $tmp -Force
            Get-ChildItem $tmp -Recurse -Include ffmpeg.exe, ffprobe.exe | Copy-Item -Destination $BinDir -Force
        } catch {
            Write-Host "ffmpeg couldn't be downloaded now; StoryStudio will offer to download it when it opens."
        } finally {
            Remove-Item $zip, $tmp -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}

# ------------------------------------------------------------------ speech model
Say "Getting the speech model (hears the words)"
$model = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"
$mdest = Join-Path $Home2 "models\$model"
if (-not (Test-Path (Join-Path $mdest "tokens.txt"))) {
    # reuse CleanView's / CleanListen's copy if this PC has one (saves a 480 MB download)
    foreach ($d in @("$env:LOCALAPPDATA\CleanView\app\models\$model", "$env:LOCALAPPDATA\CleanListen\app\models\$model")) {
        if (Test-Path (Join-Path $d "tokens.txt")) {
            New-Item -ItemType Directory -Force (Split-Path $mdest) | Out-Null
            Copy-Item $d $mdest -Recurse -Force; break
        }
    }
}
if (-not (Test-Path (Join-Path $mdest "tokens.txt"))) {
    # no other app to copy from: download StoryStudio's own copy
    $mdir = Split-Path $mdest
    New-Item -ItemType Directory -Force $mdir | Out-Null
    $tar = Join-Path $mdir "$model.tar.bz2"
    try {
        Invoke-WebRequest "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/$model.tar.bz2" -OutFile $tar
        & $vpy -c "import tarfile,sys; tarfile.open(sys.argv[1]).extractall(sys.argv[2], filter='data')" $tar $mdir
        Remove-Item $tar -Force
    } catch {
        Remove-Item $tar -Force -ErrorAction SilentlyContinue
        Write-Host "The speech model couldn't be downloaded now; StoryStudio will offer to download it when it opens."
    }
}

# ------------------------------------------------------------------ shortcuts
Say "Adding shortcuts"
$pyw = Join-Path $Venv "Scripts\pythonw.exe"
$target = Join-Path $AppDir "StoryStudio.pyw"
$ws = New-Object -ComObject WScript.Shell
$programs = [Environment]::GetFolderPath("Programs")
$desktop = [Environment]::GetFolderPath("Desktop")
foreach ($old in @((Join-Path $programs "StorySound.lnk"), (Join-Path $desktop "StorySound.lnk"))) {
    Remove-Item $old -Force -ErrorAction SilentlyContinue      # the old name
}
foreach ($l in @((Join-Path $programs "StoryStudio.lnk"), (Join-Path $desktop "StoryStudio.lnk"))) {
    $s = $ws.CreateShortcut($l)
    $s.TargetPath = $pyw
    $s.Arguments = "`"$target`""
    $s.WorkingDirectory = $AppDir
    $ico = Join-Path $AppDir "storysound.ico"
    if (Test-Path $ico) { $s.IconLocation = $ico }
    $s.Description = "StoryStudio: AI editing helper for stories, movies and music videos"
    $s.Save()
}

Say "StoryStudio $($ver.version) is installed. Open it from the Start Menu or your Desktop."
Write-Host "It keeps itself up to date: new versions install themselves when StoryStudio opens."
try { Stop-Transcript | Out-Null } catch {}
