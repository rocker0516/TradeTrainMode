"""
Launcher that starts Binance and CoinGlass services as child processes.
"""

from __future__ import annotations

import subprocess
import sys
import time
from typing import Dict, List

from services.data_fetch.config import load_config
from services.data_fetch.logging_utils import init_file_logger


logger = init_file_logger("data_fetch.launcher", "logs/data_fetch/launcher.log")


def _start_process(module_name: str) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-m", module_name])


def run_launcher() -> None:
    """Run and supervise child service processes."""
    cfg = load_config()
    modules: List[str] = [
        "services.data_fetch.binance_service",
        "services.data_fetch.coinglass_service",
    ]
    processes: Dict[str, subprocess.Popen] = {module: _start_process(module) for module in modules}
    logger.info("Started Binance + CoinGlass service processes.")

    try:
        while True:
            time.sleep(5)
            for module, process in list(processes.items()):
                if process.poll() is not None:
                    logger.warning("%s exited with code %s. Restarting...", module, process.returncode)
                    time.sleep(cfg.launcher_restart_delay_seconds)
                    processes[module] = _start_process(module)
    except KeyboardInterrupt:
        logger.info("Stopping child processes...")
        for process in processes.values():
            process.terminate()
        for process in processes.values():
            process.wait(timeout=20)
        logger.info("All services stopped.")


if __name__ == "__main__":
    run_launcher()

