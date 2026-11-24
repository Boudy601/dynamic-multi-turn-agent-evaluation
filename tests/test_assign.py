from __future__ import annotations

import sys
import unittest
from collections import Counter
import csv
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "dataset_generation"))

from assign import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    DEFAULT_ALIASES,
    DEFAULT_GUIDELINES,
    Chat,
    build_sop_resolver,
    chat_sop_id,
    distribute_snake,
    resolve_chat_guideline,
    select_initial_chats,
    select_top_up_chats,
)
from generate_coverage_report import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    write_coverage_report,
)


class AssignmentTests(unittest.TestCase):
    def test_subflow_alias_resolves_to_canonical_guideline(self) -> None:
        resolver, canonical_names, _ = build_sop_resolver(
            DEFAULT_GUIDELINES, DEFAULT_ALIASES
        )
        record = {
            "scenario": {
                "flow": "product_defect",
                "subflow": "refund_initiate",
            }
        }

        guideline = resolve_chat_guideline(record, 0, resolver)

        self.assertEqual(canonical_names[chat_sop_id(record, 0, resolver)], "Initiate Refund")
        self.assertIn("actions", guideline)

    def test_initial_selection_covers_55_sops_and_snake_is_unique(self) -> None:
        sop_ids = [f"sop_{index:02}" for index in range(55)]
        chats = [
            Chat(
                {"convo_id": f"{sop_id}_{rank}"},
                f"{sop_id}_{rank}",
                sop_id,
                100 - rank * 10 - index % 7,
                1000 - rank * 100,
            )
            for index, sop_id in enumerate(sop_ids)
            for rank in range(2)
        ]
        selected = select_initial_chats(chats, sop_ids)
        reviewers = [f"reviewer_{index}" for index in range(7)]
        assignments = distribute_snake(selected, reviewers)

        self.assertEqual(len(selected), 70)
        self.assertEqual({len(items) for items in assignments.values()}, {10})
        for items in assignments.values():
            self.assertEqual(len({chat.sop_id for chat in items}), len(items))
        self.assertEqual(
            {chat.sop_id for chat in selected},
            set(sop_ids),
        )

    def test_top_up_prefers_fresh_sop_then_updates_counts_each_pick(self) -> None:
        available = [
            Chat({}, "a-long", "sop_a", 40, 500),
            Chat({}, "b-chat", "sop_b", 30, 400),
            Chat({}, "c-chat", "sop_c", 35, 450),
        ]
        global_counts = Counter({"sop_a": 2, "sop_b": 1, "sop_c": 1})
        reviewer_counts = Counter({"sop_b": 1})

        selected = select_top_up_chats(available, global_counts, reviewer_counts, 2)

        self.assertEqual([chat.sop_id for chat in selected], ["sop_c", "sop_b"])
        self.assertEqual(global_counts, Counter({"sop_a": 2, "sop_b": 2, "sop_c": 2}))
        self.assertEqual(reviewer_counts, Counter({"sop_b": 2, "sop_c": 1}))

    def test_coverage_report_summarizes_sops_and_reviewers(self) -> None:
        first = Chat({}, "chat-1", "sop_a", 10, 100)
        second = Chat({}, "chat-2", "sop_a", 20, 200)
        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "coverage_report.csv"
            write_coverage_report(
                output_path,
                ["sop_a", "sop_b"],
                {"sop_a": "SOP A", "sop_b": "SOP B"},
                {"chat-1": first, "chat-2": second},
                {"bahaa": [first, second]},
            )
            with output_path.open(encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))

        sop_rows = [row for row in rows if row["record_type"] == "sop"]
        reviewer_rows = [row for row in rows if row["record_type"] == "reviewer"]
        self.assertEqual(len(sop_rows), 2)
        self.assertEqual(sop_rows[0]["total_chats"], "2")
        self.assertEqual(sop_rows[1]["total_chats"], "0")
        self.assertEqual(reviewer_rows[0]["total_chats"], "2")
        self.assertEqual(reviewer_rows[0]["total_messages"], "30")
        self.assertEqual(reviewer_rows[0]["average_messages"], "15.00")


if __name__ == "__main__":
    unittest.main()
