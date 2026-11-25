from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "dataset_generation"))

from assign import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    DEFAULT_ALIASES,
    DEFAULT_GUIDELINES,
    AssignmentError,
    build_sop_resolver,
    message_character_count,
)
from server import (  # noqa: E402
    ACTION_SCHEMA_FILE,
    expected_action_parameters,
    load_action_schema,
    message_list,
)


class ReviewMessageTests(unittest.TestCase):
    def test_original_dialogue_uses_delexed_action_metadata(self) -> None:
        record = {
            "original": [
                ["customer", "My ZIP code is 90210."],
                ["agent", "I will check that."],
                ["action", "Account lookup completed."],
            ],
            "delexed": [
                {
                    "speaker": "customer",
                    "text": "My ZIP code is <zip_code>.",
                    "targets": [],
                },
                {
                    "speaker": "agent",
                    "text": "I will check that.",
                    "targets": [],
                },
                {
                    "speaker": "action",
                    "text": "Account lookup completed.",
                    "targets": [
                        "account_access",
                        "take_action",
                        "pull-up-account",
                        ["customer name", "account id"],
                        -1,
                    ],
                },
            ],
        }

        messages = message_list(record)

        self.assertEqual(messages[0]["text"], "My ZIP code is 90210.")
        self.assertEqual(messages[1]["text"], "I will check that.")
        self.assertEqual(messages[2]["text"], "Account lookup completed.")
        self.assertEqual(messages[2]["function_name"], "pull-up-account")
        self.assertEqual(messages[2]["parameters"], ["customer name", "account id"])

    def test_misaligned_original_and_delexed_speakers_are_rejected(self) -> None:
        record = {
            "original": [["customer", "Hello."]],
            "delexed": [{"speaker": "agent", "text": "Hello.", "targets": []}],
        }

        with self.assertRaises(AssignmentError):
            message_list(record)

    def test_message_metrics_use_original_text(self) -> None:
        record = {
            "original": [["customer", "ZIP code 90210"]],
            "delexed": [
                {"speaker": "customer", "text": "ZIP code <zip_code>"}
            ],
        }

        self.assertEqual(message_character_count(record, 0), (1, 14))

    def test_expected_parameters_remain_scoped_to_the_subflow(self) -> None:
        resolver, _, _ = build_sop_resolver(DEFAULT_GUIDELINES, DEFAULT_ALIASES)
        schema = load_action_schema(ACTION_SCHEMA_FILE)
        record = {
            "scenario": {
                "flow": "product_defect",
                "subflow": "refund_initiate",
            }
        }

        self.assertEqual(
            expected_action_parameters(
                record,
                "Initiate Refund",
                "enter-details",
                schema,
                resolver[1],
            ),
            ["full_address", "account_id"],
        )


if __name__ == "__main__":
    unittest.main()
