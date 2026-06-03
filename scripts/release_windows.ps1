$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$DistDir = "$RepoRoot\dist"

$CudaArchitectures = "75-real;80-real;86-real;89-real;90-real;100-real;120"
$Target = if ($args.Count -gt 0) { $args[0] } else { "all" }

function Fail($msg) { Write-Error $msg; exit 1 }

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

Write-Host "architectures: $CudaArchitectures"
Write-Host "target:        $Target"
Write-Host "output:        $DistDir"
Write-Host ""

if (Test-Path $DistDir) { Remove-Item -Recurse -Force $DistDir }
New-Item -ItemType Directory -Path $DistDir | Out-Null

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
    Write-Host "info: building Python wheel"
    Push-Location $RepoRoot
    try {
        python -m build --wheel --outdir "$RepoRoot\_build_wheel" -C cmake.define.CMAKE_GENERATOR="Ninja"
        $wheel = (Get-Item "$RepoRoot\_build_wheel\*.whl").FullName
        delvewheel repair $wheel --wheel-dir "$DistDir" `
            --add-path "$RepoRoot\_build_python" `
            --exclude cublas64_13.dll `
            --exclude cublasLt64_13.dll `
            --exclude cusolver64_12.dll `
            --exclude cusparse64_12.dll
    } finally {
        Pop-Location
    }
    Write-Host ""
}

Write-Host "info: build outputs"
Get-ChildItem "$DistDir\*" -Include *.zip, *.whl | ForEach-Object {
    Write-Host "$($_.Name) ($([math]::Round($_.Length / 1MB, 1)) MB)"
}
