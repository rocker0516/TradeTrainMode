param(
    [string]$ServiceName = "TradeTrainDataFetchService",
    [string]$NssmPath = "C:\tools\nssm\nssm.exe"
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
        "-File", "`"$PSCommandPath`"",
        "-ServiceName", "`"$ServiceName`"",
        "-NssmPath", "`"$NssmPath`""
    )
    Start-Process -FilePath "powershell.exe" -Verb RunAs -ArgumentList $argList
    exit 0
}

if (-not (Test-Path $NssmPath)) {
    throw "nssm.exe not found: $NssmPath"
}

if (-not (Get-Service -Name $ServiceName -ErrorAction SilentlyContinue)) {
    Write-Host "Service does not exist: $ServiceName"
    exit 0
}

try {
    Stop-Service -Name $ServiceName -ErrorAction SilentlyContinue
} catch {
    # ignore
}

& $NssmPath remove $ServiceName confirm
Write-Host "Service removed: $ServiceName"

