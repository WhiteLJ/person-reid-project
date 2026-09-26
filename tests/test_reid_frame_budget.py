import unittest

from src.reid_frame_budget import ReIDFrameBudget


class ReIDFrameBudgetTests(unittest.TestCase):
    def test_begin_frame_resets_once_and_same_frame_is_idempotent(self) -> None:
        budget = ReIDFrameBudget(3)
        budget.begin_frame(10)
        budget.consume(2)
        budget.begin_frame(10)
        self.assertEqual(budget.used, 2)
        self.assertEqual(budget.remaining, 1)
        budget.begin_frame(11)
        self.assertEqual(budget.used, 0)
        self.assertEqual(budget.remaining, 3)

    def test_budget_rejects_overconsumption(self) -> None:
        budget = ReIDFrameBudget(1)
        budget.begin_frame(0)
        self.assertTrue(budget.can_consume())
        budget.consume()
        self.assertFalse(budget.can_consume())
        with self.assertRaises(RuntimeError):
            budget.consume()


if __name__ == "__main__":
    unittest.main()
