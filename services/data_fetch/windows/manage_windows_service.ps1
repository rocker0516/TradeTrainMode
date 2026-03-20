param(
    [ValidateSet("install", "remove")]
    [string]$Action = ""
)

$ErrorActionPreference = "Stop"

function Test-IsAdministrator {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

if (-not (Test-IsAdministrator)) {
    $argList = @(
        "-ExecutionPolicy", "Bypass",
        "-File", "`"$PSCommandPath`""
    )
    if ($Action) {
        $argList += @("-Action", $Action)
    }
    Start-Process -FilePath "powershell.exe" -Verb RunAs -ArgumentList $argList
    exit 0
}

$scriptDir = Split-Path -Parent $PSCommandPath
$installScript = Join-Path $scriptDir "install_windows_service.ps1"
$removeScript = Join-Path $scriptDir "uninstall_windows_service.ps1"

if (-not $Action) {
    Write-Host "Choose action:"
    Write-Host "1) Install service"
    Write-Host "2) Remove service"
    $choice = Read-Host "Enter 1 or 2"
    switch ($choice) {
        "1" { $Action = "install" }
        "2" { $Action = "remove" }
        default { throw "Invalid selection: $choice" }
    }
}

if ($Action -eq "install") {
    & powershell -ExecutionPolicy Bypass -File $installScript
    exit $LASTEXITCODE
}

if ($Action -eq "remove") {
    & powershell -ExecutionPolicy Bypass -File $removeScript
    exit $LASTEXITCODE
}

