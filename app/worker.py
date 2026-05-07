from __future__ import annotations

import threading

from PySide6.QtCore import QObject, Signal, Slot

from .analysis import AnalysisResult, analyze_video
from .config import AnalysisConfig


class AnalysisWorker(QObject):
    progress = Signal(int, str)
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, input_path: str, config: AnalysisConfig) -> None:
        super().__init__()
        self.input_path = input_path
        self.config = config
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    @Slot()
    def run(self) -> None:
        try:
            result: AnalysisResult = analyze_video(
                input_path=self.input_path,
                config=self.config,
                progress=lambda percent, status: self.progress.emit(percent, status),
                is_cancelled=self._cancel_event.is_set,
            )
        except RuntimeError as error:
            if "cancelled" in str(error).lower():
                self.cancelled.emit()
                return
            self.failed.emit(str(error))
            return
        except Exception as error:  # pragma: no cover - UI fallback
            self.failed.emit(str(error))
            return
        self.finished.emit(result)
