from __future__ import annotations

import unittest

import numpy as np

from src.reid_frame_cache import ReIDFrameCache


class ReIDFrameCacheTests(unittest.TestCase):
    def test_cache_is_strictly_single_frame(self) -> None:
        cache = ReIDFrameCache()
        cache.begin_frame(3)
        cache.put(7, np.asarray((1.0, 0.0), dtype=np.float32), 3)

        self.assertIsNotNone(cache.get(7, 3))
        self.assertIsNone(cache.get(7, 4))

        cache.begin_frame(4)
        self.assertIsNone(cache.get(7, 4))

    def test_cache_returns_private_embedding_copy(self) -> None:
        cache = ReIDFrameCache()
        cache.begin_frame(1)
        cache.put(7, np.asarray((1.0, 0.0), dtype=np.float32), 1)
        result = cache.get(7, 1)
        assert result is not None
        result[0] = 0.0
        self.assertEqual(float(cache.get(7, 1)[0]), 1.0)  # type: ignore[index]


if __name__ == "__main__":
    unittest.main()
