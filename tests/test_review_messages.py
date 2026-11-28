from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "dataset_generation"))

from assign import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    DEFAULT_ALIASES,
    DEFAULT_GUIDELINES,
    AssignmentError,
    build_sop_resolver,
    make_chat,
    message_character_count,
)
from server import (  # noqa: E402
    ACTION_SCHEMA_FILE,
    AppContext,
    ReviewRequestHandler,
    apply_edited_messages,
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

    def test_reviewer_chats_includes_original_messages_when_edited(self) -> None:
        resolver, canonical_names, sop_ids = build_sop_resolver(
            DEFAULT_GUIDELINES, DEFAULT_ALIASES
        )
        schema = load_action_schema(ACTION_SCHEMA_FILE)
        original_record = {
            "convo_id": 1234,
            "scenario": {
                "flow": "product_defect",
                "subflow": "refund_initiate",
            },
            "original": [
                ["customer", "My order arrived broken."],
                ["action", "Entered details."],
            ],
            "delexed": [
                {"speaker": "customer", "text": "my order arrived broken.", "targets": []},
                {
                    "speaker": "action",
                    "text": "entered details.",
                    "targets": [
                        "refund_initiate",
                        None,
                        "enter-details",
                        ["123 Main St", "acc-1"],
                        -1,
                    ],
                },
            ],
        }
        updated_record = apply_edited_messages(
            original_record,
            [
                {"speaker": "customer", "text": "My order arrived damaged."},
                {
                    "speaker": "action",
                    "text": "Entered updated details.",
                    "function_name": "enter-details",
                    "parameters": ["456 Oak Ave", "acc-2"],
                },
            ],
            edited_at="2026-11-28T12:00:00+00:00",
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assignments_dir = root / "assignments"
            reviewer_dir = assignments_dir / "alice"
            originals_dir = reviewer_dir / ".originals"
            originals_dir.mkdir(parents=True)
            reviews_file = assignments_dir / "reviews.json"

            chat_obj = make_chat(original_record, 0, resolver)
            chat_filename = f"{chat_obj.sop_id}__{chat_obj.chat_id}.json"
            chat_file = reviewer_dir / chat_filename
            chat_file.write_text(json.dumps(original_record), encoding="utf-8")

            app = AppContext(
                reviewers=["alice"],
                reviewer_sops={"alice": sop_ids[:1]},
                resolver=resolver,
                canonical_names=canonical_names,
                sop_ids=sop_ids,
                source_chats=[],
                action_schema=schema,
                assignments_dir=assignments_dir,
                reviews_file=reviews_file,
            )
            handler = ReviewRequestHandler.__new__(ReviewRequestHandler)
            handler.app = app

            unmodified = handler._reviewer_chats("alice")
            self.assertEqual(len(unmodified), 1)
            self.assertFalse(unmodified[0]["has_original"])
            self.assertIsNone(unmodified[0]["original_messages"])
            self.assertEqual(
                unmodified[0]["messages"][0]["text"], "My order arrived broken."
            )

            (originals_dir / chat_filename).write_text(
                json.dumps(original_record), encoding="utf-8"
            )
            chat_file.write_text(json.dumps(updated_record), encoding="utf-8")

            edited = handler._reviewer_chats("alice")
            self.assertEqual(len(edited), 1)
            self.assertTrue(edited[0]["has_original"])
            self.assertEqual(
                edited[0]["messages"][0]["text"], "My order arrived damaged."
            )
            self.assertIsNotNone(edited[0]["original_messages"])
            self.assertEqual(
                edited[0]["original_messages"][0]["text"],
                "My order arrived broken.",
            )
            self.assertEqual(
                edited[0]["original_messages"][1]["expected_parameters"],
                ["full_address", "account_id"],
            )


if __name__ == "__main__":
    unittest.main()
