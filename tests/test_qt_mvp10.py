from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt5.QtCore import QRectF
    from PyQt5.QtWidgets import QApplication, QMainWindow
    from src.config import load_config
    from ui.qt.gallery_page import GalleryTableModel
    from ui.qt.main_window import QtMainWindow
    from ui.qt.models import GalleryRecordDTO
    from ui.qt.widgets.video_canvas import CanvasTransform, VideoCanvas
    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False


@unittest.skipUnless(QT_AVAILABLE, "PyQt5 is an optional UI dependency")
class QtMVP10Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_letterbox_source_mapping(self) -> None:
        transform = CanvasTransform(1920, 1080, 1000, 1000, 1000 / 1920, 0, 218.75)
        point = transform.source_to_widget(960, 540)
        self.assertAlmostEqual(point[0], 500)
        self.assertAlmostEqual(point[1], 500)
        self.assertEqual(transform.widget_roi_to_source(QRectF(250, 375, 500, 250), (1080, 1920, 3)), (480, 300, 1440, 780))

    def test_gallery_model_has_only_three_columns_and_placeholder(self) -> None:
        model = GalleryTableModel()
        model.set_records((GalleryRecordDTO("person", 1, "P001", None),))
        self.assertEqual(model.columnCount(), 3)
        self.assertEqual(model.data(model.index(0, 0)), "P001")
        self.assertEqual(model.data(model.index(0, 1)), "\u65e0\u5feb\u7167")

    def test_canvas_is_widget_and_preserves_source_copy(self) -> None:
        import numpy as np

        canvas = VideoCanvas()
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        canvas.resize(1000, 1000)
        canvas.set_frame(frame)
        self.assertEqual(canvas.source_frame.shape, frame.shape)
        self.assertIsNotNone(canvas.transform)

    def test_formal_window_has_one_qmainwindow_and_four_main_controls(self) -> None:
        window = QtMainWindow(load_config("config/config.yaml"), start_worker=False)
        self.assertIsInstance(window, QMainWindow)
        self.assertEqual(
            [button.text() for button in (window.select_button, window.remove_button, window.database_button, window.quit_button)],
            ["\u6846\u9009", "\u64a4\u9500", "\u6570\u636e\u5e93", "\u9000\u51fa"],
        )
        self.assertEqual(window.stack.count(), 2)
        window._shutdown_requested = True
        window.close()


if __name__ == "__main__":
    unittest.main()
