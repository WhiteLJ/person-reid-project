from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6.QtCore import QRectF, Qt
    from PyQt6.QtWidgets import QAbstractItemView, QApplication, QMainWindow
    from src.config import load_config
    from ui.qt.appearance_log_page import AppearanceLogPage
    from ui.qt.gallery_page import GalleryTableModel
    from ui.qt.main_window import QtMainWindow
    from ui.qt.models import AppearanceLogRecordDTO, GalleryRecordDTO
    from ui.qt.widgets.video_canvas import CanvasTransform, VideoCanvas
    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False


@unittest.skipUnless(QT_AVAILABLE, "PyQt6 is an optional UI dependency")
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

    def test_gallery_model_has_plate_column_and_placeholder(self) -> None:
        model = GalleryTableModel()
        model.set_records(
            (
                GalleryRecordDTO("person", 1, "P001", None),
                GalleryRecordDTO("vehicle", 1, "V001", None, "粤B12345"),
            )
        )
        self.assertEqual(model.columnCount(), 6)
        self.assertEqual(model.data(model.index(0, 0)), "P001")
        self.assertEqual(model.data(model.index(0, 1)), "\u65e0\u5feb\u7167")
        self.assertEqual(model.data(model.index(0, 2)), "\u2014")
        self.assertEqual(model.data(model.index(0, 3)), "\u672a\u77e5")
        self.assertEqual(model.data(model.index(0, 4)), "\u5426")
        self.assertEqual(model.data(model.index(0, 5)), "\u5220\u9664")
        self.assertEqual(
            model.data(model.index(0, 4), Qt.ItemDataRole.CheckStateRole),
            Qt.CheckState.Unchecked,
        )
        self.assertTrue(
            model.setData(
                model.index(0, 4),
                Qt.CheckState.Checked,
                Qt.ItemDataRole.CheckStateRole,
            )
        )
        self.assertEqual(model.data(model.index(0, 4)), "\u662f")
        self.assertTrue(
            model.setData(
                model.index(0, 4),
                Qt.CheckState.Unchecked,
                Qt.ItemDataRole.CheckStateRole,
            )
        )
        self.assertEqual(model.data(model.index(0, 4)), "\u5426")
        model.set_records((GalleryRecordDTO("vehicle", 1, "V001", None),))
        self.assertEqual(model.data(model.index(0, 2)), "NULL")

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
        self.assertEqual(window.stack.count(), 3)
        self.assertEqual(window.select_button.font().pointSize(), 20)
        self.assertGreaterEqual(window.select_button.minimumHeight(), 64)
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
        from PyQt6.QtWidgets import QHeaderView
        from ui.qt.gallery_page import GalleryPage

        page = GalleryPage()
        self.assertTrue(hasattr(page, "import_button"))
        self.assertEqual(page.import_button.font().pointSize(), 18)
        self.assertGreaterEqual(page.import_button.minimumHeight(), 60)
        requested_domains = []
        page.importRequested.connect(requested_domains.append)
        page.import_button.click()
        page.set_domain("vehicle")
        page.import_button.click()
        self.assertEqual(requested_domains, ["person", "vehicle"])
        page.resize(1200, 800)
        page.show()
        self.app.processEvents()
        header = page.table.horizontalHeader()
        self.assertTrue(
            all(
                header.sectionResizeMode(column) == QHeaderView.ResizeMode.Stretch
                for column in range(6)
            )
        )
        self.assertGreaterEqual(page.table.font().pointSize(), 18)
        self.assertGreaterEqual(page.table.horizontalHeader().font().pointSize(), 18)
        page._toggle_batch()
        self.assertEqual(page.table.selectionMode(), QAbstractItemView.SelectionMode.MultiSelection)
        page.close()

    def test_appearance_log_page_is_read_only_and_newest_first(self) -> None:
        page = AppearanceLogPage()
        page.set_records(
            (
                AppearanceLogRecordDTO(
                    event_id=1,
                    event_time="2026-11-17 16:20:31",
                    domain="person",
                    identity_id=1,
                    display_id="P001",
                    event_type="recognition",
                    similarity=0.91234,
                    source_label="Camera 0",
                ),
                AppearanceLogRecordDTO(
                    event_id=2,
                    event_time="2026-11-17 16:31:02",
                    domain="vehicle",
                    identity_id=2,
                    display_id="V002",
                    event_type="recovery",
                    plate_number="\u7ca4B12345",
                    similarity=0.93456,
                    source_label="Camera 0",
                ),
            )
        )
        self.assertEqual(page.model.rowCount(), 2)
        self.assertEqual(page.model.data(page.model.index(0, 2)), "V002")
        self.assertEqual(page.model.data(page.model.index(0, 1)), "\u8f66\u8f86")
        self.assertEqual(page.model.data(page.model.index(0, 3)), "\u6062\u590d")
        self.assertEqual(page.model.data(page.model.index(0, 4)), "\u7ca4B12345")
        self.assertEqual(page.model.data(page.model.index(1, 1)), "\u4eba\u5458")
        self.assertEqual(page.model.data(page.model.index(1, 3)), "\u8bc6\u522b")
        self.assertEqual(page.model.data(page.model.index(1, 4)), "\u2014")

        refreshes = []
        backs = []
        page.refreshRequested.connect(lambda: refreshes.append(True))
        page.backRequested.connect(lambda: backs.append(True))
        page.refresh_button.click()
        page.back_button.click()
        self.assertEqual(refreshes, [True])
        self.assertEqual(backs, [True])
        page.close()


if __name__ == "__main__":
    unittest.main()
