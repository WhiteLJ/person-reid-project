from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt5.QtCore import QRectF
    from PyQt5.QtWidgets import QAbstractItemView, QApplication, QMainWindow
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
        window.resize(1200, 800)
        window.show()
        self.app.processEvents()
        widths = [button.width() for button in (window.select_button, window.remove_button, window.database_button, window.quit_button)]
        self.assertLessEqual(max(widths) - min(widths), 2)
        self.assertNotIn("keyPressEvent", QtMainWindow.__dict__)
        self.assertNotIn("keyPressEvent", VideoCanvas.__dict__)
        window._shutdown_requested = True
        window.close()

    def test_edit_buttons_toggle_and_modes_are_mutually_exclusive(self) -> None:
        window = QtMainWindow(load_config("config/config.yaml"), start_worker=False)
        window._edit_started("select")
        self.assertEqual(window._edit_mode, "select")
        self.assertTrue(window.select_button.isChecked())
        self.assertEqual(window.select_button.text(), "\u5b8c\u6210\u6846\u9009")
        self.assertFalse(window.remove_button.isEnabled())
        self.assertFalse(window.database_button.isEnabled())

        finish_requests = []
        window.finishEditRequested.connect(lambda: finish_requests.append(True))
        window._toggle_edit("select")
        self.assertEqual(finish_requests, [True])

        window._edit_finished()
        window._edit_started("remove")
        self.assertTrue(window.remove_button.isChecked())
        self.assertEqual(window.remove_button.text(), "\u5b8c\u6210\u64a4\u9500")
        self.assertFalse(window.select_button.isEnabled())
        self.assertFalse(window.database_button.isEnabled())
        window._edit_finished()
        self.assertEqual(window._edit_mode, None)
        self.assertTrue(window.select_button.isEnabled())
        self.assertTrue(window.remove_button.isEnabled())
        self.assertTrue(window.database_button.isEnabled())
        window._shutdown_requested = True
        window.close()

    def test_pending_save_decision_blocks_edit_finish(self) -> None:
        window = QtMainWindow(load_config("config/config.yaml"), start_worker=False)
        window._edit_started("select")
        window._popup = object()
        finish_requests = []
        window.finishEditRequested.connect(lambda: finish_requests.append(True))
        window._toggle_edit("select")
        self.assertEqual(finish_requests, [])
        self.assertTrue(window.select_button.isChecked())
        self.assertIn("\u4fdd\u5b58\u76ee\u6807", window.status_label.text())
        window._popup = None
        window._edit_finished()
        window._shutdown_requested = True
        window.close()

    def test_gallery_table_uses_full_width_and_mouse_multi_selection(self) -> None:
        from PyQt5.QtWidgets import QHeaderView
        from ui.qt.gallery_page import GalleryPage

        page = GalleryPage()
        page.resize(1200, 800)
        page.show()
        self.app.processEvents()
        header = page.table.horizontalHeader()
        self.assertTrue(all(header.sectionResizeMode(column) == QHeaderView.Stretch for column in range(3)))
        page._toggle_batch()
        self.assertEqual(page.table.selectionMode(), QAbstractItemView.MultiSelection)
        page.close()


if __name__ == "__main__":
    unittest.main()
