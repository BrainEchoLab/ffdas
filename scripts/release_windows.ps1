$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$DistDir = "$RepoRoot\dist"

$CudaMajor = if ($env:CUDA_MAJOR) { $env:CUDA_MAJOR } else { "13" }
$CudaArchitectures = if ($env:CMAKE_CUDA_ARCHITECTURES) { $env:CMAKE_CUDA_ARCHITECTURES } else { "75-real;80-real;86-real;89-real;90-real;100-real;120" }
$Target = if ($args.Count -gt 0) { $args[0] } else { "all" }

function Fail($msg) { Write-Error $msg; exit 1 }

# delvewheel excludes per CUDA major version — these are the CUDA DLLs that
# should NOT be bundled into the wheel (provided by the system toolkit or
# pip cuda packages at runtime).
if ($CudaMajor -eq "13") {
    $DelvewheelExcludes = @(
        "--exclude", "cublas64_13.dll",
        "--exclude", "cublasLt64_13.dll",
        "--exclude", "cusolver64_12.dll",
        "--exclude", "cusparse64_12.dll"
    )
} elseif ($CudaMajor -eq "12") {
    $DelvewheelExcludes = @(
        "--exclude", "cublas64_12.dll",
        "--exclude", "cublasLt64_12.dll",
        "--exclude", "cusolver64_11.dll",
        "--exclude", "cusparse64_12.dll"
    )
} else {
    Fail "unsupported CUDA_MAJOR=$CudaMajor (expected 12 or 13)"
}

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

# Validate prerequisites
if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) { Fail "cmake not found" }
if (-not (Get-Command ninja -ErrorAction SilentlyContinue)) { Fail "ninja not found" }

if ($Target -eq "python" -or $Target -eq "all") {
    if (-not (Get-Command python -ErrorAction SilentlyContinue)) { Fail "python not found" }
    if (-not (Get-Command delvewheel -ErrorAction SilentlyContinue)) { Fail "delvewheel not found (pip install delvewheel)" }
}

Write-Host "cuda major:    $CudaMajor"
Write-Host "architectures: $CudaArchitectures"
Write-Host "target:        $Target"
Write-Host "output:        $DistDir"
Write-Host ""

if (-not (Test-Path $DistDir)) { New-Item -ItemType Directory -Path $DistDir | Out-Null }

if ($Target -eq "matlab" -or $Target -eq "all") {
    Write-Host "info: building MATLAB toolbox"
    cmake -G Ninja -S $RepoRoot -B "$RepoRoot\_build_matlab" `
        -DCMAKE_BUILD_TYPE=Release `
        -DBUILD_MEX=ON `
        -DCMAKE_CUDA_ARCHITECTURES="$CudaArchitectures"
    cmake --build "$RepoRoot\_build_matlab"
    cmake --install "$RepoRoot\_build_matlab" --prefix "$DistDir\ffdas-matlab-win-x86_64"
    Compress-Archive -Path "$DistDir\ffdas-matlab-win-x86_64" -DestinationPath "$DistDir\ffdas-matlab-win-x86_64.zip"
    Write-Host ""
}

if ($Target -eq "python" -or $Target -eq "all") {
    $WheelDir = "$RepoRoot\_build_wheel_cu$CudaMajor"
    Write-Host "info: building Python wheel (ffdas-cu$CudaMajor)"
    $env:FFDAS_CUDA_MAJOR = $CudaMajor
    $env:CMAKE_CUDA_ARCHITECTURES = $CudaArchitectures
    $env:CMAKE_GENERATOR = "Ninja"
    Push-Location $RepoRoot
    try {
        if (Test-Path $WheelDir) { Remove-Item -Recurse -Force $WheelDir }
        python -m build --wheel --outdir $WheelDir
        $wheel = (Get-Item "$WheelDir\*.whl").FullName
        & delvewheel repair $wheel --wheel-dir "$DistDir" @DelvewheelExcludes
    } finally {
        Pop-Location
    }
    Write-Host ""
}

Write-Host "info: build outputs"
Get-ChildItem "$DistDir\*" -Include *.zip, *.whl | ForEach-Object {
    Write-Host "$($_.Name) ($([math]::Round($_.Length / 1MB, 1)) MB)"
}
