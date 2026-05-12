from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import pyqtgraph as pg
from PySide6.QtCore import QPoint, QThread, Qt, QUrl
from PySide6.QtGui import QDesktopServices, QDragEnterEvent, QDropEvent, QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QDoubleSpinBox,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QScrollArea,
    QFrame,
    QVBoxLayout,
    QWidget,
    QToolTip,
)

from .analysis import AnalysisResult, FrameResult, encode_thumbnail
from .config import AnalysisConfig
from .worker import AnalysisWorker


def frame_to_pixmap(frame_rgb) -> QPixmap:
    height, width, channels = frame_rgb.shape
    bytes_per_line = channels * width
    image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(image.copy())


class MetricTrendsWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Metric trends")
        self.resize(900, 520)
        self._frames: list[FrameResult] = []

        layout = QVBoxLayout(self)

        controls = QWidget()
        controls_layout = QHBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        self.chart_x_mode = QComboBox()
        self.chart_x_mode.addItems(["Frame index (timeline)", "Rank (sorted)"])
        self.chart_x_mode.currentIndexChanged.connect(self._render)
        self.chart_connect_lines = QCheckBox("Connect points")
        self.chart_connect_lines.setChecked(False)
        self.chart_connect_lines.toggled.connect(self._render)
        controls_layout.addWidget(QLabel("X-axis:"))
        controls_layout.addWidget(self.chart_x_mode)
        controls_layout.addSpacing(12)
        controls_layout.addWidget(self.chart_connect_lines)
        controls_layout.addStretch(1)

        self.plot_widget = pg.PlotWidget(title="Metric trends")
        self.plot_widget.addLegend()
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setLabel("left", "Score")
        self.plot_widget.setLabel("bottom", "Frame index")
        self.plot_widget.setYRange(0, 100)

        layout.addWidget(controls)
        layout.addWidget(self.plot_widget, 1)

    def set_frames(self, frames: list[FrameResult]) -> None:
        self._frames = frames
        self._render()

    def _render(self) -> None:
        self.plot_widget.clear()
        if self.plot_widget.plotItem.legend is not None:
            self.plot_widget.plotItem.legend.clear()
        if not self._frames:
            return

        use_rank = self.chart_x_mode.currentIndex() == 1
        ordered = sorted(self._frames, key=lambda item: item.rank if use_rank else item.frame_index)
        x_values = [item.rank if use_rank else item.frame_index for item in ordered]
        x_label = "Rank" if use_rank else "Frame index"
        self.plot_widget.setLabel("bottom", x_label)

        connect_lines = self.chart_connect_lines.isChecked()
        line_pen_final = pg.mkPen("w", width=1.5) if connect_lines else None
        line_pen_metric = pg.mkPen((120, 120, 120, 120), width=1) if connect_lines else None

        self.plot_widget.plot(
            x_values,
            [item.final for item in ordered],
            pen=line_pen_final,
            symbol="o",
            symbolSize=6,
            symbolBrush=pg.mkBrush("w"),
            symbolPen=None,
            name="final",
        )
        self.plot_widget.plot(
            x_values,
            [item.sharpness for item in ordered],
            pen=line_pen_metric,
            symbol="o",
            symbolSize=5,
            symbolBrush=pg.mkBrush("c"),
            symbolPen=None,
            name="sharpness",
        )
        self.plot_widget.plot(
            x_values,
            [item.exposure for item in ordered],
            pen=line_pen_metric,
            symbol="o",
            symbolSize=5,
            symbolBrush=pg.mkBrush("y"),
            symbolPen=None,
            name="exposure",
        )
        self.plot_widget.plot(
            x_values,
            [item.face for item in ordered],
            pen=None,
            symbol="t",
            symbolSize=8,
            symbolBrush=pg.mkBrush("m"),
            symbolPen=None,
            name="face",
        )
        self.plot_widget.setYRange(0, 100)


class HoverHintLabel(QLabel):
    def enterEvent(self, event):  # noqa: N802 - Qt API
        tooltip = self.toolTip()
        if tooltip:
            label_pos = self.mapToGlobal(self.rect().topLeft())
            text_y = int((self.rect().height() - self.fontMetrics().height()) / 2)
            QToolTip.showText(label_pos + QPoint(120, text_y - 15), tooltip, self)
        super().enterEvent(event)


class NumericTableWidgetItem(QTableWidgetItem):
    def __init__(self, text: str, numeric_value: float) -> None:
        super().__init__(text)
        self.numeric_value = numeric_value

    def __lt__(self, other):  # noqa: D401 - Qt comparator API
        if isinstance(other, NumericTableWidgetItem):
            return self.numeric_value < other.numeric_value
        return super().__lt__(other)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Best Frame Analyzer")
        self.resize(1500, 900)
        self.setAcceptDrops(True)

        self.current_input_path: str | None = None
        self.current_output_dir = "out"
        self.current_result: AnalysisResult | None = None
        self.current_preview_frame = None
        self._thread: QThread | None = None
        self._worker: AnalysisWorker | None = None
        self._default_config = AnalysisConfig()
        self.trends_window = MetricTrendsWindow()
        self._is_drag_active = False
        self._initial_layout_applied = False

        self._build_ui()
        self._set_running(False)

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QVBoxLayout(root)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(4)

        main_layout.addWidget(self._build_top_bar())

        left_panel_scroll = QScrollArea()
        left_panel_scroll.setWidgetResizable(True)
        left_panel_scroll.setFrameShape(QFrame.Shape.NoFrame)
        left_panel_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_panel_scroll.setWidget(self._build_left_panel())

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.addWidget(left_panel_scroll)
        self.main_splitter.addWidget(self._build_results_panel())
        self.main_splitter.setStretchFactor(1, 1)
        main_layout.addWidget(self.main_splitter, 1)

        self.drop_hint = QLabel("Drop MP4 file here", root)
        self.drop_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drop_hint.setVisible(False)
        self.drop_hint.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.drop_hint.setStyleSheet(
            "QLabel {"
            "background-color: rgba(20, 20, 20, 180);"
            "border: 2px dashed #66ccff;"
            "border-radius: 10px;"
            "color: #e8f7ff;"
            "font-size: 24px;"
            "font-weight: 600;"
            "padding: 24px;"
            "}"
        )
        self.drop_hint.raise_()
        self.drop_hint.setGeometry(root.rect().adjusted(30, 30, -30, -30))

        footer = QWidget()
        footer_layout = QVBoxLayout(footer)
        footer_layout.setContentsMargins(8, 0, 8, 0)
        footer_layout.setSpacing(2)

        self.status_label = QLabel("Drop MP4 file or select one.")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        footer_layout.addWidget(self.status_label)
        footer_layout.addWidget(self.progress)
        main_layout.addWidget(footer)

        self._apply_responsive_layout()

    def _build_top_bar(self) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(8, 2, 8, 2)
        layout.setSpacing(6)

        self.input_line = QLineEdit()
        self.input_line.setReadOnly(True)
        self.output_line = QLineEdit(self.current_output_dir)

        self.pick_input_button = QPushButton("Select MP4")
        self.pick_input_button.clicked.connect(self._pick_input)
        self.pick_output_button = QPushButton("Select Output")
        self.pick_output_button.clicked.connect(self._pick_output_dir)

        self.start_button = QPushButton("Start Analysis")
        self.start_button.clicked.connect(self._start_analysis)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self._cancel_analysis)
        self.open_output_button = QPushButton("Open Output Folder")
        self.open_output_button.clicked.connect(self._open_output_folder)
        self.show_trends_button = QPushButton("Show Metric Trends")
        self.show_trends_button.clicked.connect(self._show_trends_window)

        layout.addWidget(QLabel("Input:"))
        layout.addWidget(self.input_line, 2)
        layout.addWidget(self.pick_input_button)
        layout.addWidget(QLabel("Output:"))
        layout.addWidget(self.output_line, 1)
        layout.addWidget(self.pick_output_button)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.show_trends_button)
        layout.addWidget(self.open_output_button)
        return widget

    def _build_left_panel(self) -> QWidget:
        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)

        params_group = QGroupBox("Parameters")
        params_layout = QFormLayout(params_group)

        self.step_spin = QSpinBox()
        self.step_spin.setRange(1, 10_000)
        self.step_spin.setValue(self._default_config.step)

        self.w_sharpness = self._make_double_spin(0.0, 1.0, self._default_config.w_sharpness, step=0.01)
        self.w_exposure = self._make_double_spin(0.0, 1.0, self._default_config.w_exposure, step=0.01)
        self.w_face = self._make_double_spin(0.0, 1.0, self._default_config.w_face, step=0.01)
        self.ear_threshold = self._make_double_spin(0.0, 1.0, self._default_config.ear_threshold, step=0.01)
        self.overexpose_threshold = self._make_double_spin(0.0, 1.0, self._default_config.overexpose_threshold, step=0.01)
        self.underexpose_threshold = self._make_double_spin(0.0, 1.0, self._default_config.underexpose_threshold, step=0.01)
        self.overexpose_pixel = QSpinBox()
        self.overexpose_pixel.setRange(0, 255)
        self.overexpose_pixel.setValue(self._default_config.overexpose_pixel_value)
        self.underexpose_pixel = QSpinBox()
        self.underexpose_pixel.setRange(0, 255)
        self.underexpose_pixel.setValue(self._default_config.underexpose_pixel_value)
        self.optimal_min = self._make_double_spin(0.0, 255.0, self._default_config.optimal_brightness_min, step=1.0)
        self.optimal_max = self._make_double_spin(0.0, 255.0, self._default_config.optimal_brightness_max, step=1.0)
        self.ratio_penalty_scale = self._make_double_spin(0.0, 5000.0, self._default_config.ratio_penalty_scale, step=10.0)
        self.brightness_distance_scale = self._make_double_spin(
            0.0, 10.0, self._default_config.brightness_distance_scale, step=0.05
        )
        self._set_parameter_tooltips()

        params_layout.addRow(self._param_label("Frame step", self.step_spin.toolTip()), self.step_spin)
        params_layout.addRow(self._param_label("Weight sharpness", self.w_sharpness.toolTip()), self.w_sharpness)
        params_layout.addRow(self._param_label("Weight exposure", self.w_exposure.toolTip()), self.w_exposure)
        params_layout.addRow(self._param_label("Weight face", self.w_face.toolTip()), self.w_face)
        params_layout.addRow(self._param_label("EAR threshold", self.ear_threshold.toolTip()), self.ear_threshold)
        params_layout.addRow(
            self._param_label("Overexpose threshold", self.overexpose_threshold.toolTip()),
            self.overexpose_threshold,
        )
        params_layout.addRow(
            self._param_label("Underexpose threshold", self.underexpose_threshold.toolTip()),
            self.underexpose_threshold,
        )
        params_layout.addRow(self._param_label("Overexpose pixel", self.overexpose_pixel.toolTip()), self.overexpose_pixel)
        params_layout.addRow(
            self._param_label("Underexpose pixel", self.underexpose_pixel.toolTip()),
            self.underexpose_pixel,
        )
        params_layout.addRow(self._param_label("Optimal brightness min", self.optimal_min.toolTip()), self.optimal_min)
        params_layout.addRow(self._param_label("Optimal brightness max", self.optimal_max.toolTip()), self.optimal_max)
        params_layout.addRow(
            self._param_label("Ratio penalty scale", self.ratio_penalty_scale.toolTip()),
            self.ratio_penalty_scale,
        )
        params_layout.addRow(
            self._param_label("Brightness distance scale", self.brightness_distance_scale.toolTip()),
            self.brightness_distance_scale,
        )

        reset_button = QPushButton("Reset defaults")
        reset_button.clicked.connect(self._reset_defaults)
        params_layout.addRow(reset_button)

        input_group = QGroupBox("Input video")
        input_layout = QVBoxLayout(input_group)
        input_top = QWidget()
        input_top_layout = QHBoxLayout(input_top)
        input_top_layout.setContentsMargins(0, 0, 0, 0)
        input_top_layout.setSpacing(6)
        self.input_video_name = QLabel("No file selected")
        self.input_video_name.setWordWrap(True)
        self.clear_input_button = QPushButton("X")
        self.clear_input_button.setFixedWidth(28)
        self.clear_input_button.setToolTip("Remove selected input video")
        self.clear_input_button.clicked.connect(self._clear_input_path)
        input_top_layout.addWidget(self.input_video_name, 1)
        input_top_layout.addWidget(self.clear_input_button)

        self.input_preview = QLabel("No preview")
        self.input_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.input_preview.setMinimumHeight(160)
        self.input_preview.setStyleSheet("border: 1px solid #555;")
        self.input_add_button = QPushButton("+", self.input_preview)
        self.input_add_button.setToolTip("Select MP4 file")
        self.input_add_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.input_add_button.setFlat(True)
        self.input_add_button.setStyleSheet(
            "QPushButton {"
            "color: rgba(170, 170, 170, 160);"
            "font-size: 42px;"
            "font-weight: 300;"
            "border: none;"
            "background: transparent;"
            "padding: 0px;"
            "}"
            "QPushButton:hover {"
            "color: rgba(200, 200, 200, 210);"
            "}"
        )
        self.input_add_button.clicked.connect(self._pick_input)

        layout.addWidget(params_group)
        input_layout.addWidget(input_top)
        input_layout.addWidget(self.input_preview)
        layout.addWidget(input_group)
        layout.addStretch(1)
        self._update_input_card()
        return wrapper

    @staticmethod
    def _param_label(text: str, tooltip: str) -> QLabel:
        label = HoverHintLabel(text)
        label.setToolTip(tooltip)
        return label

    def _set_parameter_tooltips(self) -> None:
        self.step_spin.setToolTip("Analyze every N-th frame. Higher value is faster but less precise.")
        self.w_sharpness.setToolTip("Weight of sharpness metric in final score (0..1).")
        self.w_exposure.setToolTip("Weight of exposure metric in final score (0..1).")
        self.w_face.setToolTip("Weight of face/eyes metric in final score (0..1).")
        self.ear_threshold.setToolTip("EAR threshold for open eyes. Higher value makes detection stricter.")
        self.overexpose_threshold.setToolTip("Allowed ratio of very bright pixels before overexposure penalty.")
        self.underexpose_threshold.setToolTip("Allowed ratio of very dark pixels before underexposure penalty.")
        self.overexpose_pixel.setToolTip("Pixel intensity treated as overexposed (0..255).")
        self.underexpose_pixel.setToolTip("Pixel intensity treated as underexposed (0..255).")
        self.optimal_min.setToolTip("Lower bound of preferred mean brightness range.")
        self.optimal_max.setToolTip("Upper bound of preferred mean brightness range.")
        self.ratio_penalty_scale.setToolTip("Strength of over/underexposure ratio penalty.")
        self.brightness_distance_scale.setToolTip("Penalty per brightness unit outside optimal range.")

    def _build_results_panel(self) -> QWidget:
        wrapper = QWidget()
        layout = QHBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.best_preview = QLabel("Best frame preview")
        self.best_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.best_preview.setMinimumHeight(420)
        self.best_preview.setStyleSheet("border: 1px solid #555;")
        self.best_preview.setMinimumWidth(700)
        self.export_preview_button = QPushButton("Export preview image")
        self.export_preview_button.setEnabled(False)
        self.export_preview_button.clicked.connect(self._export_current_preview)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)
        left_layout.addWidget(self.best_preview, 1)
        left_layout.addWidget(self.export_preview_button, 0, Qt.AlignmentFlag.AlignRight)

        self.frames_table = QTableWidget(0, 6)
        self.frames_table.setHorizontalHeaderLabels(["Rank", "Frame", "Final", "Sharpness", "Exposure", "Face"])
        self.frames_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.frames_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.frames_table.itemSelectionChanged.connect(self._sync_from_table_selection)
        self.frames_table.verticalHeader().setVisible(False)
        self.frames_table.setSortingEnabled(True)
        self.frames_table.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frames_table.horizontalHeader().setStretchLastSection(False)
        self.frames_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        self._adjust_table_columns()

        self.thumbnail_list = QListWidget()
        self.thumbnail_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.thumbnail_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.thumbnail_list.setIconSize(QPixmap(180, 100).size())
        self.thumbnail_list.setMovement(QListWidget.Movement.Static)
        self.thumbnail_list.currentRowChanged.connect(self._sync_from_thumbnail_selection)

        self.right_panel = QWidget()
        right_layout = QVBoxLayout(self.right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)
        right_layout.addWidget(self.frames_table, 2)
        right_layout.addWidget(self.thumbnail_list, 1)

        self.results_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.results_splitter.addWidget(left_panel)
        self.results_splitter.addWidget(self.right_panel)
        self.results_splitter.setStretchFactor(0, 3)
        self.results_splitter.setStretchFactor(1, 2)

        layout.addWidget(self.results_splitter)
        return wrapper

    @staticmethod
    def _make_double_spin(min_value: float, max_value: float, value: float, step: float) -> QDoubleSpinBox:
        widget = QDoubleSpinBox()
        widget.setRange(min_value, max_value)
        widget.setDecimals(4)
        widget.setSingleStep(step)
        widget.setValue(value)
        return widget

    def _reset_defaults(self) -> None:
        defaults = self._default_config
        self.step_spin.setValue(defaults.step)
        self.w_sharpness.setValue(defaults.w_sharpness)
        self.w_exposure.setValue(defaults.w_exposure)
        self.w_face.setValue(defaults.w_face)
        self.ear_threshold.setValue(defaults.ear_threshold)
        self.overexpose_threshold.setValue(defaults.overexpose_threshold)
        self.underexpose_threshold.setValue(defaults.underexpose_threshold)
        self.overexpose_pixel.setValue(defaults.overexpose_pixel_value)
        self.underexpose_pixel.setValue(defaults.underexpose_pixel_value)
        self.optimal_min.setValue(defaults.optimal_brightness_min)
        self.optimal_max.setValue(defaults.optimal_brightness_max)
        self.ratio_penalty_scale.setValue(defaults.ratio_penalty_scale)
        self.brightness_distance_scale.setValue(defaults.brightness_distance_scale)

    def _build_config(self) -> AnalysisConfig:
        config = replace(
            self._default_config,
            step=self.step_spin.value(),
            output_dir=self.output_line.text().strip() or "out",
            w_sharpness=self.w_sharpness.value(),
            w_exposure=self.w_exposure.value(),
            w_face=self.w_face.value(),
            ear_threshold=self.ear_threshold.value(),
            overexpose_threshold=self.overexpose_threshold.value(),
            underexpose_threshold=self.underexpose_threshold.value(),
            overexpose_pixel_value=self.overexpose_pixel.value(),
            underexpose_pixel_value=self.underexpose_pixel.value(),
            optimal_brightness_min=self.optimal_min.value(),
            optimal_brightness_max=self.optimal_max.value(),
            ratio_penalty_scale=self.ratio_penalty_scale.value(),
            brightness_distance_scale=self.brightness_distance_scale.value(),
        )
        config.validate()
        return config

    def _set_running(self, running: bool) -> None:
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.pick_input_button.setEnabled(not running)
        self.pick_output_button.setEnabled(not running)

    def _pick_input(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(self, "Select MP4", "", "Video files (*.mp4)")
        if selected:
            self._set_input_path(selected)

    def _pick_output_dir(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Select output directory", self.output_line.text() or "out")
        if selected:
            self.output_line.setText(selected)

    def _set_input_path(self, path: str) -> None:
        self.current_input_path = path
        self.input_line.setText(path)
        self.status_label.setText(f"Ready: {path}")
        self._update_input_card()

    def _clear_input_path(self) -> None:
        self.current_input_path = None
        self.input_line.clear()
        self._update_input_card()
        self.status_label.setText("Drop MP4 file or select one.")

    def _update_input_card(self) -> None:
        has_input = bool(self.current_input_path)
        self.clear_input_button.setEnabled(has_input)
        self._position_input_add_button()
        if not has_input:
            self.input_video_name.setText("No file selected")
            self.input_preview.setText("")
            self.input_preview.setPixmap(QPixmap())
            self.input_add_button.show()
            return

        input_path = Path(self.current_input_path)
        self.input_video_name.setText(input_path.name)
        self.input_add_button.hide()
        preview_pixmap = self._extract_input_preview_pixmap(str(input_path))
        if preview_pixmap is None:
            self.input_preview.setText("Preview unavailable")
            self.input_preview.setPixmap(QPixmap())
            return
        fitted = preview_pixmap.scaled(
            self.input_preview.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.input_preview.setPixmap(fitted)

    @staticmethod
    def _extract_input_preview_pixmap(path: str) -> QPixmap | None:
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            return None

        try:
            if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
                capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
            ok, frame = capture.read()
            if not ok or frame is None:
                return None
            thumb_rgb = encode_thumbnail(frame, width=280)
            return frame_to_pixmap(thumb_rgb)
        finally:
            capture.release()

    def _start_analysis(self) -> None:
        if not self.current_input_path:
            QMessageBox.warning(self, "Missing input", "Select MP4 file first.")
            return
        try:
            config = self._build_config()
        except Exception as error:
            QMessageBox.critical(self, "Invalid settings", str(error))
            return

        self.progress.setValue(0)
        self.status_label.setText("Starting analysis...")

        self._thread = QThread(self)
        self._worker = AnalysisWorker(self.current_input_path, config)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.failed.connect(self._on_failed)
        self._worker.cancelled.connect(self._on_cancelled)
        self._worker.finished.connect(self._on_finished)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._worker.cancelled.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_thread)

        self._set_running(True)
        self._thread.start()

    def _cancel_analysis(self) -> None:
        if self._worker:
            self._worker.cancel()
            self.status_label.setText("Cancelling...")

    def _cleanup_thread(self) -> None:
        self._worker = None
        self._thread = None
        self._set_running(False)
        self.show_trends_button.setEnabled(True)

    def _on_progress(self, percent: int, status: str) -> None:
        self.progress.setValue(percent)
        self.status_label.setText(status)

    def _on_failed(self, message: str) -> None:
        self.status_label.setText("Analysis failed")
        QMessageBox.critical(self, "Analysis error", message)

    def _on_cancelled(self) -> None:
        self.status_label.setText("Analysis cancelled")
        self.progress.setValue(0)

    def _on_finished(self, result: AnalysisResult) -> None:
        self.current_result = result
        self.progress.setValue(100)
        self.status_label.setText(f"Done. Best frame index={result.best_frame.frame_index}, score={result.best_frame.final:.2f}")
        self._render_results(result)

    def _render_results(self, result: AnalysisResult) -> None:
        self._set_preview_from_frame(result.best_frame.frame)

        self.frames_table.setSortingEnabled(False)
        self.frames_table.setRowCount(len(result.frames))
        for row, item in enumerate(result.frames):
            values = [
                (str(item.rank), float(item.rank)),
                (str(item.frame_index), float(item.frame_index)),
                (f"{item.final:.2f}", float(item.final)),
                (f"{item.sharpness:.2f}", float(item.sharpness)),
                (f"{item.exposure:.2f}", float(item.exposure)),
                (f"{item.face:.2f}", float(item.face)),
            ]
            for col, (label, numeric_value) in enumerate(values):
                cell = NumericTableWidgetItem(label, numeric_value)
                cell.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                cell.setData(Qt.ItemDataRole.UserRole, item.frame_index)
                self.frames_table.setItem(row, col, cell)
        self.frames_table.setSortingEnabled(True)
        self._adjust_table_columns()
        self.frames_table.sortByColumn(0, Qt.SortOrder.AscendingOrder)

        self.thumbnail_list.clear()
        for item in result.frames:
            thumb_rgb = encode_thumbnail(item.frame, width=180)
            pixmap = frame_to_pixmap(thumb_rgb)
            list_item = QListWidgetItem(f"#{item.rank} f{item.frame_index} ({item.final:.1f})")
            list_item.setIcon(pixmap)
            self.thumbnail_list.addItem(list_item)

        self.trends_window.set_frames(result.frames)
        if result.frames:
            self.frames_table.selectRow(0)
            self.thumbnail_list.setCurrentRow(0)

    def _sync_from_table_selection(self) -> None:
        if self.current_result is None:
            return
        selected_rows = self.frames_table.selectionModel().selectedRows()
        if not selected_rows:
            return
        row = selected_rows[0].row()
        frame_item = self.frames_table.item(row, 1)
        if frame_item is None:
            return
        frame_index = int(frame_item.text())
        thumb_row = self._thumbnail_row_for_frame_index(frame_index)
        if thumb_row >= 0 and self.thumbnail_list.currentRow() != thumb_row:
            self.thumbnail_list.setCurrentRow(thumb_row)

    def _sync_from_thumbnail_selection(self, row: int) -> None:
        if self.current_result is None or row < 0 or row >= len(self.current_result.frames):
            return
        item = self.current_result.frames[row]
        self._set_preview_from_frame(item.frame)
        self.status_label.setText(f"Selected rank {item.rank}, frame {item.frame_index}, score {item.final:.2f}")
        table_row = self._table_row_for_frame_index(item.frame_index)
        if table_row >= 0 and self.frames_table.currentRow() != table_row:
            self.frames_table.selectRow(table_row)

    def _open_output_folder(self) -> None:
        output_path = Path(self.output_line.text().strip() or "out").resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(output_path)))

    def _show_trends_window(self) -> None:
        if self.current_result is None:
            QMessageBox.information(self, "No data", "Run analysis first to view metric trends.")
            return
        self.trends_window.set_frames(self.current_result.frames)
        self.trends_window.show()
        self.trends_window.raise_()
        self.trends_window.activateWindow()

    def _export_current_preview(self) -> None:
        if self.current_preview_frame is None:
            QMessageBox.information(self, "No preview", "No preview image to export.")
            return

        suggested = "preview_frame.png"
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save preview image",
            suggested,
            "PNG image (*.png);;JPEG image (*.jpg *.jpeg);;BMP image (*.bmp)",
        )
        if not save_path:
            return

        ok = cv2.imwrite(save_path, self.current_preview_frame)
        if not ok:
            QMessageBox.critical(self, "Save failed", f"Could not save file: {save_path}")
            return
        self.status_label.setText(f"Saved preview image: {save_path}")

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802 - Qt API
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if any(url.isLocalFile() and url.toLocalFile().lower().endswith(".mp4") for url in urls):
                event.acceptProposedAction()
                self._set_drag_feedback(True)
                return
        event.ignore()

    def dragMoveEvent(self, event):  # noqa: N802 - Qt API
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if any(url.isLocalFile() and url.toLocalFile().lower().endswith(".mp4") for url in urls):
                event.acceptProposedAction()
                self._set_drag_feedback(True)
                return
        self._set_drag_feedback(False)
        event.ignore()

    def dragLeaveEvent(self, event):  # noqa: N802 - Qt API
        self._set_drag_feedback(False)
        event.accept()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802 - Qt API
        self._set_drag_feedback(False)
        for url in event.mimeData().urls():
            if url.isLocalFile():
                local = url.toLocalFile()
                if local.lower().endswith(".mp4"):
                    self._set_input_path(local)
                    event.acceptProposedAction()
                    return
        event.ignore()

    def resizeEvent(self, event):  # noqa: N802 - Qt API
        super().resizeEvent(event)
        if self.centralWidget() is not None:
            self.drop_hint.setGeometry(self.centralWidget().rect().adjusted(30, 30, -30, -30))
        self._apply_responsive_layout()
        self._position_input_add_button()
        self._update_input_card()
        if self.current_result is not None:
            row = self.thumbnail_list.currentRow()
            if row < 0:
                row = 0
            if 0 <= row < len(self.current_result.frames):
                self._set_preview_from_frame(self.current_result.frames[row].frame)

    def showEvent(self, event):  # noqa: N802 - Qt API
        super().showEvent(event)
        if not self._initial_layout_applied:
            # Ensure first visible layout already matches responsive proportions.
            self._apply_responsive_layout()
            self._initial_layout_applied = True

    def _set_drag_feedback(self, active: bool) -> None:
        if self._is_drag_active == active:
            return
        self._is_drag_active = active
        self.drop_hint.setVisible(active)
        if active:
            self.status_label.setText("Release to import MP4")
        elif self.current_input_path:
            self.status_label.setText(f"Ready: {self.current_input_path}")
        else:
            self.status_label.setText("Drop MP4 file or select one.")

    def _apply_responsive_layout(self) -> None:
        width = max(900, self.width())

        # Keep top bar compact and visually attached to title bar.
        top_button_width = max(110, int(width * 0.07))
        for button in (
            self.pick_input_button,
            self.pick_output_button,
            self.start_button,
            self.stop_button,
            self.show_trends_button,
            self.open_output_button,
        ):
            button.setMinimumWidth(top_button_width)

        # Scale preview and thumbnails with window width.
        self.best_preview.setMinimumWidth(max(320, int(width * 0.30)))
        thumb_w = max(120, int(width * 0.10))
        thumb_h = max(70, int(thumb_w * 0.56))
        self.thumbnail_list.setIconSize(QPixmap(thumb_w, thumb_h).size())
        self._adjust_table_columns()

        # Keep right panel wide enough for table/thumbnail content (no clipping).
        if hasattr(self, "results_splitter"):
            total_results = max(1, self.results_splitter.width())
            min_preview_target = int(width * 0.30)
            desired_right = self._desired_right_panel_width()
            self.right_panel.setMinimumWidth(desired_right)

            max_right_with_min_preview = max(1, total_results - min_preview_target)
            if desired_right <= max_right_with_min_preview:
                right_target = desired_right
                preview_target = total_results - right_target
            else:
                # If both constraints cannot be satisfied, prefer no clipping in the table.
                right_target = min(total_results - 1, desired_right)
                preview_target = max(1, total_results - right_target)
            self.results_splitter.setSizes([preview_target, right_target])

        # Keep global proportions stable: parameters 26%, results 74%.
        if hasattr(self, "main_splitter"):
            total = max(1, self.main_splitter.width())
            self.main_splitter.setSizes([int(total * 0.26), int(total * 0.74)])

    def _position_input_add_button(self) -> None:
        button_size = 64
        self.input_add_button.setFixedSize(button_size, button_size)
        x = max(0, (self.input_preview.width() - button_size) // 2)
        y = max(0, (self.input_preview.height() - button_size) // 2)
        self.input_add_button.move(x, y)

    def _set_preview_from_frame(self, frame) -> None:
        self.current_preview_frame = frame.copy()
        self.export_preview_button.setEnabled(True)
        target_width = max(200, self.best_preview.width() - 20)
        rgb = encode_thumbnail(frame, width=target_width)
        pixmap = frame_to_pixmap(rgb)
        fitted = pixmap.scaled(
            self.best_preview.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.best_preview.setPixmap(fitted)

    def _thumbnail_row_for_frame_index(self, frame_index: int) -> int:
        if self.current_result is None:
            return -1
        for idx, item in enumerate(self.current_result.frames):
            if item.frame_index == frame_index:
                return idx
        return -1

    def _table_row_for_frame_index(self, frame_index: int) -> int:
        for row in range(self.frames_table.rowCount()):
            item = self.frames_table.item(row, 1)
            if item is None:
                continue
            if item.text() == str(frame_index):
                return row
        return -1

    def _adjust_table_columns(self) -> None:
        header = self.frames_table.horizontalHeader()
        fm = header.fontMetrics()
        width_multiplier = 1.18  # small percentage offset over header text width
        base_padding = 18
        for col in range(self.frames_table.columnCount()):
            label = self.frames_table.horizontalHeaderItem(col).text()
            content_width = fm.horizontalAdvance(label)
            target = int((content_width + base_padding) * width_multiplier)
            self.frames_table.setColumnWidth(col, max(68, target))

    def _desired_right_panel_width(self) -> int:
        table_columns_width = 0
        if self.frames_table.verticalHeader().isVisible():
            table_columns_width += self.frames_table.verticalHeader().width()
        for col in range(self.frames_table.columnCount()):
            table_columns_width += self.frames_table.columnWidth(col)
        table_columns_width += self.frames_table.frameWidth() * 2
        if self.frames_table.verticalScrollBar().isVisible():
            table_columns_width += self.frames_table.verticalScrollBar().sizeHint().width()
        table_columns_width += 12  # light safety padding

        thumbnail_min = self.thumbnail_list.iconSize().width() + 52
        return max(table_columns_width, thumbnail_min)
