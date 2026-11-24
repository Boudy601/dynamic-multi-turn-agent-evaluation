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
    initialize_assignments,
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

    def test_initial_selection_bundles_pairs_and_balances_reviewers(self) -> None:
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
        self.assertEqual(
            Counter(Counter(chat.sop_id for chat in selected).values()),
            Counter({1: 40, 2: 15}),
        )
        self.assertEqual({len(items) for items in assignments.values()}, {10})
        reviewer_sop_counts = {
            reviewer: Counter(chat.sop_id for chat in items)
            for reviewer, items in assignments.items()
        }
        self.assertEqual(
            Counter(len(counts) for counts in reviewer_sop_counts.values()),
            Counter({7: 1, 8: 6}),
        )
        for sop_id in sop_ids:
            owners = [
                reviewer
                for reviewer, counts in reviewer_sop_counts.items()
                if counts[sop_id]
            ]
            self.assertEqual(len(owners), 1)
        self.assertEqual(
            {chat.sop_id for chat in selected},
            set(sop_ids),
        )

    def test_initialize_replaces_assignments_and_clears_old_reviews(self) -> None:
        sop_ids = [f"sop_{index:02}" for index in range(55)]
        chats = [
            Chat(
                {"convo_id": f"{sop_id}_{rank}"},
                f"{sop_id}_{rank}",
                sop_id,
                100 - rank,
                1000 - rank,
            )
            for sop_id in sop_ids
            for rank in range(2)
        ]
        reviewers = [f"reviewer_{index}" for index in range(7)]

        with TemporaryDirectory() as temp_dir:
            assignments_dir = Path(temp_dir) / "assignments"
            old_reviewer_dir = assignments_dir / "reviewer_0"
            old_reviewer_dir.mkdir(parents=True)
            (old_reviewer_dir / "old_chat.json").write_text("{}", encoding="utf-8")
            (assignments_dir / "reviews.json").write_text(
                '{"old_chat": "error"}', encoding="utf-8"
            )
            (assignments_dir / "coverage_report.csv").write_text(
                "stale report", encoding="utf-8"
            )

            assignments = initialize_assignments(
                chats, reviewers, assignments_dir, sop_ids
            )

            assigned_files = [
                path
                for reviewer_dir in assignments_dir.iterdir()
                if reviewer_dir.is_dir()
                for path in reviewer_dir.glob("*.json")
            ]
            self.assertEqual(sum(map(len, assignments.values())), 70)
            self.assertEqual(len(assigned_files), 70)
            self.assertFalse((assignments_dir / "reviews.json").exists())
            self.assertFalse((assignments_dir / "coverage_report.csv").exists())
            self.assertFalse((old_reviewer_dir / "old_chat.json").exists())

    def test_top_up_prefers_familiar_sop_and_updates_counts_each_pick(self) -> None:
        available = [
            Chat({}, "a-familiar", "sop_a", 20, 200),
            Chat({}, "b-chat", "sop_b", 30, 400),
            Chat({}, "c-chat", "sop_c", 35, 450),
        ]
        global_counts = Counter({"sop_a": 1, "sop_b": 1, "sop_c": 1})
        reviewer_counts = Counter({"sop_a": 1})

        selected = select_top_up_chats(available, global_counts, reviewer_counts, 2)

        self.assertEqual([chat.sop_id for chat in selected], ["sop_a", "sop_c"])
        self.assertEqual(global_counts, Counter({"sop_a": 2, "sop_b": 1, "sop_c": 2}))
        self.assertEqual(reviewer_counts, Counter({"sop_a": 2, "sop_c": 1}))

    def test_global_coverage_precedes_reviewer_familiarity(self) -> None:
        available = [
            Chat({}, "familiar", "sop_familiar", 100, 1000),
            Chat({}, "under-covered", "sop_under", 20, 200),
        ]
        global_counts = Counter({"sop_familiar": 2, "sop_under": 1})
        reviewer_counts = Counter({"sop_familiar": 1})

        selected = select_top_up_chats(available, global_counts, reviewer_counts, 1)

        self.assertEqual([chat.sop_id for chat in selected], ["sop_under"])

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
