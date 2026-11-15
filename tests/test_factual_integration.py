import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "dataset_generation"))

import factual_hallucination as factual  # noqa: E402
from prepare_prompts import flatten_guidelines, load_judges, prepare_records  # noqa: E402
from generate_golden_dataset import build_factual_output_record  # noqa: E402


class FactualIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.judges = load_judges(
            json.loads((PROJECT_ROOT / "config/judges_definitions.json").read_text(encoding="utf-8")),
            PROJECT_ROOT / "config/judges_definitions.json",
        )
        cls.guidelines = flatten_guidelines(
            json.loads((PROJECT_ROOT / "data/source/abcd/data/guidelines.json").read_text(encoding="utf-8")),
            PROJECT_ROOT / "data/source/abcd/data/guidelines.json",
        )
        cls.template = PROJECT_ROOT / "prompts/factual_hallucination_injection_prompt_template.txt"

    def _conversation_with_agent_claim(self):
        # based on a real ABCD conversation structure but with an explicit Agent factual claim
        return {
            "convo_id": 999001,
            "scenario": {
                "personal": {"customer_name": "joyce wu", "phone": "(186) 372-1256"},
                "order": {"order_id": "2621786777"},
                "flow": "subscription_inquiry",
                "subflow": "manage_extension",
            },
            "original": [
                ["agent", "Hello, how can i help you?"],
                ["customer", "Can I delay my subscription payment?"],
                ["agent", "I have found your account Joyce Wu. I will now need your account id and order id to verify your identity"],
                ["customer", "Account ID: XRU6MBJHP6. Order ID: 2621786777"],
            ],
        }

    def test_prepare_factual_prompt_for_eligible_conversation(self):
        conv = self._conversation_with_agent_claim()
        import random

        records = prepare_records(
            conversations=[conv],
            guidelines=self.guidelines,
            judges=self.judges,
            requested_judges=["factual_hallucination"],
            conversation_count=1,
            rng=random.Random(0),
            prompt_template_path=self.template,
        )
        self.assertEqual(len(records), 1)
        rec = records[0]
        self.assertEqual(rec["target_judge"], "factual_hallucination")
        self.assertIn("assembled_prompt", rec)
        self.assertIn("source_conversation", rec)
        self.assertIn("available_evidence", rec)
        self.assertIn("judge_definition", rec)
        self.assertIn("FACTUAL HALLUCINATION", rec["assembled_prompt"])
        self.assertIn("eligible candidate", rec["assembled_prompt"].lower())
        self.assertIn("joyce wu", rec["assembled_prompt"])

    def test_prepare_factual_skips_ineligible_conversations(self):
        sample = json.loads((PROJECT_ROOT / "data/source/abcd/data/abcd_sample.json").read_text(encoding="utf-8"))
        conv = sample[0]  # known to have no eligible agent-stated factual claim
        import random

        records = prepare_records(
            conversations=[conv],
            guidelines=self.guidelines,
            judges=self.judges,
            requested_judges=["factual_hallucination"],
            conversation_count=1,
            rng=random.Random(0),
            prompt_template_path=self.template,
        )
        self.assertEqual(records, [])

    def test_factual_validation_and_output_shape(self):
        conv = self._conversation_with_agent_claim()
        turns = factual.extract_conversation_turns(conv)
        scenario = conv["scenario"]
        evidence = factual.filter_supported_evidence(
            factual.build_available_evidence(scenario), scenario
        )
        ctx = factual.build_factual_validation_context(
            conversation_id=conv["convo_id"],
            source_conversation=turns,
            scenario=scenario,
            sop={},
            judge_definition=self.judges["factual_hallucination"],
            available_evidence=evidence,
        )
        original_claim = "your account Joyce Wu"
        injected_claim = "your account John Wu"
        payload = {
            "injection_metadata": {
                "target_judge": "factual_hallucination",
                "already_defective": False,
                "injection_status": "injected",
                "injection_turn_id": 3,
                "scope": "single_turn",
                "error_category": "customer_identity",
                "error_classification": "CONTRADICTED",
                "original_agent_claim": original_claim,
                "injected_claim": injected_claim,
                "evidence_source_type": "scenario",
                "evidence_source_location": "scenario.personal.customer_name",
                "evidence_text": "joyce wu",
                "contradiction_explanation": "Agent states John wu while trusted name is joyce wu.",
                "scenario_modified": False,
                "modified_scenario_fields": {},
                "non_target_risk_assessment": {
                    "action_accuracy": False,
                    "procedure_following": False,
                    "intent_understanding": False,
                    "task_closure": False,
                    "alternative_guidance": False,
                    "missing_data": False,
                    "data_conflict": False,
                    "notes": "Name-only text change.",
                },
                "expected_target_verdict": "FAIL",
            },
            "injected_conversation": [
                {**t, "content": t["content"].replace(original_claim, injected_claim) if t["turn_id"] == 3 else t["content"]}
                for t in turns
            ],
        }
        rec = factual.validate_factual_golden_record(payload, {}, ctx)
        out = build_factual_output_record(conv["convo_id"], rec, ctx)
        self.assertEqual(
            set(out.keys()),
            {
                "conversation_id",
                "injection_metadata",
                "injected_conversation",
                "judge_evaluations",
                "audit_context",
                "validation_report",
            },
        )
        self.assertEqual(out["audit_context"]["source_conversation"], turns)
        self.assertEqual(out["audit_context"]["source_scenario"], scenario)
        self.assertEqual(
            out["audit_context"]["available_evidence"], evidence
        )
        self.assertEqual(
            out["validation_report"]["evidence_verification"]["classification"],
            "CONTRADICTED",
        )
        self.assertEqual(out["injection_metadata"]["target_judge"], "factual_hallucination")
        self.assertIn("factual_hallucination", out["judge_evaluations"])
        self.assertEqual(out["judge_evaluations"]["factual_hallucination"]["verdict"], "FAIL")
        self.assertEqual(len(out["injected_conversation"]), len(turns))
        self.assertNotEqual(out["injected_conversation"][2]["content"], turns[2]["content"])
        self.assertEqual(out["injected_conversation"][0]["content"], turns[0]["content"])
        self.assertEqual(out["injected_conversation"][1]["content"], turns[1]["content"])

    def test_rejection_detection(self):
        payload = {"injection_status": "INJECTION_NOT_POSSIBLE", "rejection_code": "NO_TRUSTED_EVIDENCE", "rejection_reason": "x"}
        self.assertTrue(factual.is_injection_not_possible(payload, json.dumps(payload)))
        self.assertFalse(factual.is_injection_not_possible({"injection_metadata": {"injection_status": "injected"}}, ""))

    def test_prepare_prompts_cli_batch_selection(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "prepared.jsonl"
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "src/dataset_generation/prepare_prompts.py"),
                "--batch", "factual_hallucination",
                "--output", str(out),
                "--overwrite",
                "--allow-duplicates",
                "--seed", "7",
            ]
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, env=env)
            self.assertEqual(proc.returncode, 0, proc.stdout + "\n" + proc.stderr)
            # may produce 0 or more records depending on eligible candidates; must be valid jsonl
            if out.exists():
                lines = [ln for ln in out.read_text(encoding="utf-8").splitlines() if ln.strip()]
                for ln in lines:
                    obj = json.loads(ln)
                    self.assertEqual(obj["target_judge"], "factual_hallucination")


if __name__ == "__main__":
    unittest.main()
