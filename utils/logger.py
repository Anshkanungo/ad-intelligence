"""
Ad Intelligence Pipeline — Structured Logger

Provides consistent logging across all modules.
Usage:
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Processing image", extra={"module": "ocr"})
"""

import logging
import sys
import time
from functools import wraps
from typing import Any, Callable


# ══════════════════════════════════════════════
# CUSTOM FORMATTER — Clean, readable logs
# ══════════════════════════════════════════════

class PipelineFormatter(logging.Formatter):
    """Color-coded, structured log formatter."""

    COLORS = {
        "DEBUG":    "\033[90m",     # gray
        "INFO":     "\033[94m",     # blue
        "WARNING":  "\033[93m",     # yellow
        "ERROR":    "\033[91m",     # red
        "CRITICAL": "\033[91m\033[1m",  # bold red
    }
    RESET = "\033[0m"
    ICONS = {
        "DEBUG":    "🔍",
        "INFO":     "ℹ️ ",
        "WARNING":  "⚠️ ",
        "ERROR":    "❌",
        "CRITICAL": "🔥",
    }

    def format(self, record: logging.LogRecord) -> str:
        level = record.levelname
        color = self.COLORS.get(level, "")
        icon = self.ICONS.get(level, "")
        reset = self.RESET

        # Module name extraction
        module = getattr(record, "module_name", record.module)

        timestamp = time.strftime("%H:%M:%S", time.localtime(record.created))
        msg = f"{color}{icon} [{timestamp}] [{module}] {record.getMessage()}{reset}"

        if record.exc_info and record.exc_info[0]:
            msg += f"\n{self.formatException(record.exc_info)}"

        return msg


# ══════════════════════════════════════════════
# LOGGER FACTORY
# ══════════════════════════════════════════════

_loggers: dict[str, logging.Logger] = {}


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Get a configured logger for a module.

    Args:
        name: Module name (usually __name__)
        level: Log level string (DEBUG, INFO, WARNING, ERROR)

    Returns:
        Configured logger instance
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(f"ad_intelligence.{name}")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    # Only add handler if none exist
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(PipelineFormatter())
        logger.addHandler(handler)

    _loggers[name] = logger
    return logger


# ══════════════════════════════════════════════
# TIMING DECORATOR — Track module execution time
# ══════════════════════════════════════════════

def log_execution_time(func: Callable) -> Callable:
    """
    Decorator that logs how long a function takes.
    Use on pipeline module entry points.

    Usage:
        @log_execution_time
        def run_ocr(image):
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        logger = get_logger(func.__module__)
        func_name = func.__qualname__
        logger.info(f"Starting: {func_name}")
        start = time.perf_counter()

        try:
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logger.info(f"Completed: {func_name} ({elapsed:.2f}s)")
            return result
        except Exception as e:
            elapsed = time.perf_counter() - start
            logger.error(f"Failed: {func_name} ({elapsed:.2f}s) — {type(e).__name__}: {e}")
            raise

    return wrapper


# ══════════════════════════════════════════════
# PIPELINE PROGRESS LOGGER
# ══════════════════════════════════════════════

class PipelineProgress:
    """
    Tracks overall pipeline progress for UI updates.

    Usage:
        progress = PipelineProgress(total_steps=6)
        progress.step("Running OCR...")
        progress.step("Running YOLO...")
        progress.complete()
    """

    def __init__(self, total_steps: int = 1):
        self.total_steps = total_steps
        self.current_step = 0
        self.steps_log: list[dict] = []
        self.start_time = time.perf_counter()
        self.logger = get_logger("pipeline.progress")

    def step(self, message: str):
        """Log a pipeline step."""
        self.current_step += 1
        elapsed = time.perf_counter() - self.start_time
        step_info = {
            "step": self.current_step,
            "total": self.total_steps,
            "message": message,
            "elapsed_sec": round(elapsed, 2),
        }
        self.steps_log.append(step_info)
        self.logger.info(f"[{self.current_step}/{self.total_steps}] {message}")

    def complete(self) -> dict:
        """Mark pipeline as complete, return summary."""
        elapsed = time.perf_counter() - self.start_time
        summary = {
            "total_steps": self.current_step,
            "total_time_sec": round(elapsed, 2),
            "steps": self.steps_log,
        }
        self.logger.info(f"Pipeline complete — {self.current_step} steps in {elapsed:.2f}s")
        return summary

    @property
    def progress_pct(self) -> float:
        """Current progress as percentage (0-100)."""
        if self.total_steps == 0:
            return 100.0
        return round((self.current_step / self.total_steps) * 100, 1)


# ══════════════════════════════════════════════
# QUICK TEST
# ══════════════════════════════════════════════

if __name__ == "__main__":
    logger = get_logger("test", level="DEBUG")
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning")
    logger.error("This is an error")

    # Test timing decorator
    @log_execution_time
    def fake_ocr():
        time.sleep(0.1)
        return ["text1", "text2"]

    result = fake_ocr()
    print(f"  Result: {result}")

    # Test progress tracker
    progress = PipelineProgress(total_steps=3)
    progress.step("Running OCR...")
    time.sleep(0.05)
    progress.step("Running YOLO...")
    time.sleep(0.05)
    progress.step("Running LLM...")
    summary = progress.complete()
    print(f"  Progress: {progress.progress_pct}%")
    print(f"  Summary: {summary}")