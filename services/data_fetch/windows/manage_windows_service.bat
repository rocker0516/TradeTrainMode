@echo off
setlocal
cd /d C:\Users\User\Desktop\python\TradeTrainMode
powershell -ExecutionPolicy Bypass -File "services\data_fetch\windows\manage_windows_service.ps1"
endlocal

