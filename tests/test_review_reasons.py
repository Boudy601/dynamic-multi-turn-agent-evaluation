from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "dataset_generation"))

from server import (  # noqa: E402
    ApiError,
    load_reviews,
    save_reviews,
    validate_review_entry,
)


class ReviewReasonTests(unittest.TestCase):
    def test_load_reviews_preserves_legacy_statuses_and_new_reasons(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reviews.json"
            path.write_text(
                json.dumps(
                    {
                        "reviewer": {
                            "legacy-chat": "error",
                            "reasoned-chat": {
                                "status": "error",
                                "reason": "The action used the wrong account.",
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )

            reviews = load_reviews(path)

        self.assertEqual(
            reviews["reviewer"]["legacy-chat"],
            {"status": "error", "reason": ""},
        )
        self.assertEqual(
            reviews["reviewer"]["reasoned-chat"],
            {
                "status": "error",
                "reason": "The action used the wrong account.",
            },
        )

    def test_save_reviews_persists_reason(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reviews.json"
            save_reviews(
                path,
                {
                    "reviewer": {
                        "chat": {
                            "status": "error",
                            "reason": "Incorrect refund amount.",
                        }
                    }
                },
            )

            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8"))["reviewer"]["chat"],
                {"status": "error", "reason": "Incorrect refund amount."},
            )

    def test_load_reviews_rejects_oversized_reason(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reviews.json"
            path.write_text(
                json.dumps(
                    {
                        "reviewer": {
                            "chat": {"status": "error", "reason": "x" * 301}
                        }
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(ApiError):
                load_reviews(path)

    def test_error_status_requires_short_reason(self) -> None:
        self.assertEqual(
            validate_review_entry("error", "  Incorrect refund amount.  "),
            {"status": "error", "reason": "Incorrect refund amount."},
        )
        self.assertEqual(
            validate_review_entry("no_error"),
            {"status": "no_error", "reason": ""},
        )
        with self.assertRaises(ApiError):
            validate_review_entry("error", "  ")
        with self.assertRaises(ApiError):
            validate_review_entry("error", "x" * 301)


if __name__ == "__main__":
    unittest.main()
