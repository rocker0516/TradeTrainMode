@echo off
setlocal
cd /d C:\Users\User\Desktop\python\TradeTrainMode

REM 如果你的 python 不在 PATH，請改成完整路徑，例如:
REM "C:\Users\User\anaconda3\python.exe" -m services.data_fetch.launcher
python -m services.data_fetch.launcher

endlocal

