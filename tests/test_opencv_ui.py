from __future__ import annotations

import unittest

from ui.opencv_ui import UIAction, key_to_action


class OpenCVUIActionTests(unittest.TestCase):
    def test_key_mapping(self) -> None:
        self.assertEqual(key_to_action(ord("s")), UIAction.SELECT_TARGET)
        self.assertEqual(key_to_action(ord("R")), UIAction.REMOVE_TARGET)
        self.assertEqual(key_to_action(ord("c")), UIAction.CLEAR_TARGETS)
        self.assertEqual(key_to_action(ord("g")), UIAction.ENROLL_GALLERY)
        self.assertEqual(key_to_action(ord("Q")), UIAction.QUIT)
        self.assertEqual(key_to_action(-1), UIAction.NONE)


if __name__ == "__main__":
    unittest.main()
