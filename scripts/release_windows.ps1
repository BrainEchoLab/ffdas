$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$DistDir = "$RepoRoot\dist"

$CudaRoot = if ($env:CUDA_ROOT) { $env:CUDA_ROOT } else { $null }
$Target = if ($args.Count -gt 0) { $args[0] } else { "all" }
$ffdasVersion = "0.1.0"

function Fail($msg) { Write-Error $msg; exit 1 }

# Validate prerequisites
if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) { Fail "cmake not found" }
if (-not (Get-Command ninja -ErrorAction SilentlyContinue)) { Fail "ninja not found" }

# Detect CUDA root from environment or PATH
if (-not $CudaRoot) {
    $nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
    if ($nvcc) {
        $CudaRoot = Split-Path -Parent (Split-Path -Parent $nvcc.Source)
    } else {
        Fail "CUDA_ROOT not set and nvcc not on PATH"
    }
}

$nvccPath = "$CudaRoot\bin\nvcc.exe"
if (-not (Test-Path $nvccPath)) { Fail "nvcc not found at $nvccPath" }

# Detect CUDA major version
$nvccOutput = & $nvccPath --version 2>&1 | Out-String
if ($nvccOutput -match "release (\d+)") {
    $CudaMajor = $Matches[1]
} else {
    Fail "could not detect CUDA major version from nvcc"
}
Write-Host "info: detected CUDA major version: $CudaMajor"

if ($CudaMajor -eq "13") {
    $CudaArchitectures = "75-real;80-real;86-real;89-real;90-real;100-real;120"
} elseif ($CudaMajor -eq "12") {
    $CudaArchitectures = "75-real;80-real;86-real;89-real;90"
} else {
    Fail "unsupported CUDA major version $CudaMajor (expected 12 or 13)"
}

# delvewheel excludes — CUDA DLLs provided by the system or pip
$DelvewheelExcludes = @(
    "--exclude", "cublas64_*.dll",
    "--exclude", "cublasLt64_*.dll",
    "--exclude", "cusolver64_*.dll",
    "--exclude", "cusparse64_*.dll"
)

# Set up VS developer environment if not already active
if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    $vswhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) { Fail "vswhere.exe not found; is Visual Studio installed?" }
    $vsPath = & $vswhere -latest -property installationPath
    if (-not $vsPath) { Fail "no Visual Studio installation found" }
    $devShell = "$vsPath\Common7\Tools\Microsoft.VisualStudio.DevShell.dll"
    if (-not (Test-Path $devShell)) { Fail "DevShell module not found at $devShell" }
    Import-Module $devShell
    Enter-VsDevShell -VsInstallPath $vsPath -Arch amd64 -SkipAutomaticLocation
    Write-Host "info: loaded VS developer environment from $vsPath"
}

if ($Target -eq "python" -or $Target -eq "all") {
    if (-not (Get-Command python -ErrorAction SilentlyContinue)) { Fail "python not found" }
    if (-not (Get-Command delvewheel -ErrorAction SilentlyContinue)) { Fail "delvewheel not found (pip install delvewheel)" }
}

Write-Host "info: CUDA_ROOT=$CudaRoot"
Write-Host "info: CUDA_ARCHITECTURES=$CudaArchitectures"
Write-Host "info: TARGET=$Target"
Write-Host "info: DIST_DIR=$DistDir"
Write-Host ""

if (-not (Test-Path $DistDir)) { New-Item -ItemType Directory -Path $DistDir | Out-Null }

# Stage 1: build the core library
$LibBuildDir = "$RepoRoot\_build_lib_cu$CudaMajor"
Write-Host "info: building ffdas_cu$CudaMajor"
cmake -G Ninja -S $RepoRoot -B $LibBuildDir `
    -DCMAKE_BUILD_TYPE=Release `
    -DCUDAToolkit_ROOT="$CudaRoot" `
    -DCMAKE_CUDA_ARCHITECTURES="$CudaArchitectures"
cmake --build $LibBuildDir

if (-not (Test-Path "$LibBuildDir\ffdas_cu$CudaMajor.dll")) {
    Fail "ffdas_cu$CudaMajor.dll not found in $LibBuildDir"
}

# Stage 2a: MATLAB toolbox
if ($Target -eq "matlab" -or $Target -eq "all") {
    $MatlabBuildDir = "$RepoRoot\_build_matlab_cu$CudaMajor"
    $MatlabDistName = "ffdas_cu$CudaMajor-$ffdasVersion-matlab-win_amd64"
    Write-Host ""
    Write-Host "info: building MATLAB toolbox"
    cmake -G Ninja -S "$RepoRoot\bindings\matlab" -B $MatlabBuildDir `
        -DCMAKE_BUILD_TYPE=Release `
        -DFFDAS_LIB_DIR="$LibBuildDir" `
        -DFFDAS_INCLUDE_DIR="$RepoRoot\include"
    cmake --build $MatlabBuildDir
    cmake --install $MatlabBuildDir --prefix "$DistDir\$MatlabDistName"
    Compress-Archive -Path "$DistDir\$MatlabDistName" -DestinationPath "$DistDir\$MatlabDistName.zip"
    Write-Host ""
}

# Stage 2b: Python wheel
if ($Target -eq "python" -or $Target -eq "all") {
    $WheelDir = "$RepoRoot\_build_wheel_cu$CudaMajor"
    Write-Host ""
    Write-Host "info: building Python wheel (ffdas-cu$CudaMajor)"
    $env:FFDAS_LIB_DIR = $LibBuildDir
    $env:CUDA_ROOT = $CudaRoot
    $env:CMAKE_GENERATOR = "Ninja"
    Push-Location $RepoRoot
    try {
        if (Test-Path $WheelDir) { Remove-Item -Recurse -Force $WheelDir }
        python -m build --wheel --outdir $WheelDir
        $wheel = (Get-Item "$WheelDir\*.whl").FullName
        & delvewheel repair $wheel --wheel-dir "$DistDir" --add-path "$LibBuildDir" @DelvewheelExcludes
    } finally {
        Pop-Location
    }
    Write-Host ""
}

Write-Host "info: build outputs"
Get-ChildItem "$DistDir\*" -Include *.zip, *.whl | ForEach-Object {
    Write-Host "$($_.Name) ($([math]::Round($_.Length / 1MB, 1)) MB)"
}
