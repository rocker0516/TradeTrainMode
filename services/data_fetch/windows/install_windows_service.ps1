param(
    [string]$ServiceName = "TradeTrainDataFetchService",
    [string]$NssmPath = "C:\tools\nssm\nssm.exe",
    [string]$PythonExe = "",
    [string]$ProjectRoot = "C:\Users\User\Desktop\python\TradeTrainMode",
    [bool]$AutoStart = $true
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
        "-NssmPath", "`"$NssmPath`"",
        "-PythonExe", "`"$PythonExe`"",
        "-ProjectRoot", "`"$ProjectRoot`""
    )
    Start-Process -FilePath "powershell.exe" -Verb RunAs -ArgumentList $argList
    exit 0
}

if (-not (Test-Path $NssmPath)) {
    throw "nssm.exe not found: $NssmPath"
}

if (-not $PythonExe) {
    $projectCondaPython = Join-Path $ProjectRoot ".conda\python.exe"
    $projectCondaScriptsPython = Join-Path $ProjectRoot ".conda\Scripts\python.exe"

    if (Test-Path $projectCondaPython) {
        $PythonExe = $projectCondaPython
    } elseif (Test-Path $projectCondaScriptsPython) {
        $PythonExe = $projectCondaScriptsPython
    } else {
        $pythonCommand = Get-Command "python" -ErrorAction SilentlyContinue
        if ($null -eq $pythonCommand) {
            throw "Python executable not found. Please pass -PythonExe explicitly."
        }
        $PythonExe = $pythonCommand.Source
    }
} else {
    $pythonCommand = Get-Command $PythonExe -ErrorAction SilentlyContinue
    if ($null -ne $pythonCommand) {
        $PythonExe = $pythonCommand.Source
    }
}

if (-not (Test-Path $PythonExe)) {
    throw "Python executable path does not exist: $PythonExe"
}

$appParameters = "-m services.data_fetch.launcher"

if (Get-Service -Name $ServiceName -ErrorAction SilentlyContinue) {
    & $NssmPath stop $ServiceName
    & $NssmPath remove $ServiceName confirm
}

& $NssmPath install $ServiceName $PythonExe $appParameters
& $NssmPath set $ServiceName AppDirectory $ProjectRoot
& $NssmPath set $ServiceName Start SERVICE_AUTO_START
& $NssmPath set $ServiceName DisplayName "TradeTrain Data Fetch Service"
& $NssmPath set $ServiceName Description "Runs Binance and CoinGlass data fetch loops."

if (-not (Get-Service -Name $ServiceName -ErrorAction SilentlyContinue)) {
    throw "Service installation verification failed: $ServiceName"
}

Write-Host "Service installed: $ServiceName"
Write-Host "Python executable: $PythonExe"

if ($AutoStart) {
    Start-Service -Name $ServiceName
    Start-Sleep -Seconds 8
    $service = Get-Service -Name $ServiceName -ErrorAction Stop
    if ($service.Status -ne "Running") {
        throw "Service start verification failed: $ServiceName is $($service.Status)"
    }

    $launcher = Get-CimInstance Win32_Process -Filter "name='python.exe'" | Where-Object {
        $_.CommandLine -match "services\.data_fetch\.launcher"
    }
    $binance = Get-CimInstance Win32_Process -Filter "name='python.exe'" | Where-Object {
        $_.CommandLine -match "services\.data_fetch\.binance_service"
    }
    $coinglass = Get-CimInstance Win32_Process -Filter "name='python.exe'" | Where-Object {
        $_.CommandLine -match "services\.data_fetch\.coinglass_service"
    }

    Write-Host "Service running: $ServiceName"
    Write-Host "Launcher process count: $($launcher.Count)"
    Write-Host "Binance process count: $($binance.Count)"
    Write-Host "CoinGlass process count: $($coinglass.Count)"
}

