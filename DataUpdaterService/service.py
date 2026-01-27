"""Windows Service entry (pywin32).

用法：
  python -m DataUpdaterService.service --install
  python -m DataUpdaterService.service --start
  python -m DataUpdaterService.service --stop
  python -m DataUpdaterService.service --remove

注意：此模組只在 Windows + 安裝 pywin32 時可用。
"""

from __future__ import annotations

import platform
from datetime import datetime, timedelta
from pathlib import Path

from DataUpdaterService.config import DataUpdaterConfig
from DataUpdaterService.runner import DataUpdaterRunner


def _project_root_from_this_file() -> Path:
    # DataUpdaterService/ 位於專案根目錄下
    return Path(__file__).resolve().parent.parent


def _next_daily_run(now: datetime, *, hhmm) -> datetime:
    """計算下一次每日執行時間（local time）。"""
    target = now.replace(hour=hhmm.hour, minute=hhmm.minute, second=0, microsecond=0)
    if target <= now:
        target = target + timedelta(days=1)
    return target


def _ensure_windows() -> None:
    if platform.system().lower() != "windows":
        raise RuntimeError("DataUpdaterService.service is only available on Windows.")


def _run_service_main() -> None:
    _ensure_windows()

    import win32event  # type: ignore
    import win32service  # type: ignore
    import win32serviceutil  # type: ignore
    import servicemanager  # type: ignore

    class DataUpdaterWindowsService(win32serviceutil.ServiceFramework):
        _svc_name_ = "TradeTrainModeDataUpdater"
        _svc_display_name_ = "TradeTrainMode Data Updater"
        _svc_description_ = "Auto update Data/*.csv (5m every N seconds, 1d daily) for TradeTrainMode."

        def __init__(self, args):
            super().__init__(args)
            self.stop_event = win32event.CreateEvent(None, 0, 0, None)

            project_root = _project_root_from_this_file()
            cfg = DataUpdaterConfig.from_config_file(project_root=project_root)
            self.runner = DataUpdaterRunner(config=cfg, project_root=project_root)

            self.update_5m_seconds = max(30, int(cfg.update_5m_seconds))
            self.update_1d_time = cfg.update_1d_time
            self.next_5m = datetime.now()
            self.next_1d = _next_daily_run(datetime.now(), hhmm=self.update_1d_time)

        def SvcStop(self):
            self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
            win32event.SetEvent(self.stop_event)

        def SvcDoRun(self):
            servicemanager.LogMsg(
                servicemanager.EVENTLOG_INFORMATION_TYPE,
                servicemanager.PYS_SERVICE_STARTED,
                (self._svc_name_, ""),
            )
            self.main()

        def main(self):
            # simple scheduler loop
            while True:
                rc = win32event.WaitForSingleObject(self.stop_event, 1000)
                if rc == win32event.WAIT_OBJECT_0:
                    break

                now = datetime.now()
                if now >= self.next_5m:
                    try:
                        self.runner.run_5m_once()
                    except Exception as e:
                        servicemanager.LogErrorMsg(f"5m update failed: {e}")
                    self.next_5m = now + timedelta(seconds=self.update_5m_seconds)

                if now >= self.next_1d:
                    try:
                        self.runner.run_1d_once()
                    except Exception as e:
                        servicemanager.LogErrorMsg(f"1d update failed: {e}")
                    self.next_1d = _next_daily_run(now, hhmm=self.update_1d_time)

            servicemanager.LogMsg(
                servicemanager.EVENTLOG_INFORMATION_TYPE,
                servicemanager.PYS_SERVICE_STOPPED,
                (self._svc_name_, ""),
            )

    win32serviceutil.HandleCommandLine(DataUpdaterWindowsService)


def main() -> None:
    """Module entrypoint."""
    _run_service_main()


if __name__ == "__main__":
    main()


