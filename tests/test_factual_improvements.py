import json
import random
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "dataset_generation"))

import factual_hallucination as factual  # noqa: E402
from prepare_prompts import (  # noqa: E402
    flatten_guidelines,
    load_judges,
    prepare_records,
    load_prompt_template,
    resolve_prompt_template_path,
    get_template_placeholders,
    prepare_single_record,
)
from generate_golden_dataset import GoldenRecordOutput  # noqa: E402


def _status_scenario(status="delivered"):
    return {
        "personal": {"customer_name": "Jane Doe"},
        "order": {"order_id": "12345", "shipping_status": status},
        "flow": "product_defect",
        "subflow": "return_size",
    }


class FactualImprovementTests(unittest.TestCase):
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
        cls.factual_template = PROJECT_ROOT / "prompts/factual_hallucination_injection_prompt_template.txt"

    def _select(self, turns, scenario, sop=None):
        evidence = factual.filter_supported_evidence(
            factual.build_available_evidence(scenario), scenario
        )
        return factual.select_factual_candidates(turns, scenario, evidence, sop=sop)

    def test_conditional_or_generic_status_rule_excluded(self):
        scenario = _status_scenario("delivered")
        turns = [
            {"turn_id": 1, "speaker": "AGENT", "content": "If the order is out for delivery, it has already left our facility."},
        ]
        # make evidence match a concrete value that appears? "out for delivery" maps to in_transit; evidence says delivered.
        # generic rule should be excluded even if it mentions statuses not matching evidence; create case matching evidence in context.
        turns = [
            {"turn_id": 1, "speaker": "AGENT", "content": "If an order is delivered, it has arrived at your address."},
            {"turn_id": 2, "speaker": "CUSTOMER", "content": "Thanks."},
        ]
        sel = self._select(turns, scenario, sop=None)
        self.assertEqual(len(sel.eligible), 0)
        codes = {e.reason_code for e in sel.exclusions}
        self.assertIn("CONDITIONAL_OR_GENERIC_CLAIM", codes)

    def test_clear_pre_existing_non_target_defect_causes_rejection(self):
        # Explicit, unambiguous contradiction between ACTION outcome and AGENT claim.
        conv = {
            "convo_id": 9001,
            "scenario": {
                "personal": {"customer_name": "Jane Doe"},
                "order": {"shipping_status": "delivered"},
                "flow": "product_defect",
                "subflow": "return_size",
            },
            "original": [
                ["customer", "Please process my refund."],
                ["action", "Refund failed due to payment error."],
                ["agent", "Your refund was completed successfully."],
            ],
        }
        turns = factual.extract_conversation_turns(conv)
        clear, _ = factual._assess_pre_existing_non_target_defects(
            turns, conv["scenario"], {}, factual.build_available_evidence(conv["scenario"])
        )
        self.assertTrue(clear)

        records = prepare_records(
            conversations=[conv],
            guidelines=self.guidelines,
            judges=self.judges,
            requested_judges=["factual_hallucination"],
            conversation_count=1,
            rng=random.Random(0),
            prompt_template_path=self.factual_template,
        )
        self.assertEqual(records, [])

    def test_missing_action_log_evidence_alone_does_not_automatically_reject(self):
        # Eligible claim should remain eligible even if SOP exists but no ACTION proves steps;
        # absence of action log must not create a clear defect rejection.
        scenario = {
            "personal": {"customer_name": "Joyce Wu"},
            "order": {"order_id": "2621786777"},
            "flow": "subscription_inquiry",
            "subflow": "manage_extension",
        }
        sop = {"steps": ["verify identity"]}
        turns = [
            {"turn_id": 1, "speaker": "AGENT", "content": "I have found your account Joyce Wu."},
            {"turn_id": 2, "speaker": "CUSTOMER", "content": "Thanks."},
        ]
        sel = self._select(turns, scenario, sop=sop)
        self.assertGreaterEqual(len(sel.eligible), 1)
        clear, review = factual._assess_pre_existing_non_target_defects(
            turns, scenario, sop, factual.build_available_evidence(scenario)
        )
        self.assertEqual(clear, [])
        # may have human review flags due to uncertainty, but not an automatic rejection

    def test_replacement_that_changes_sop_branch_is_rejected(self):
        scenario = _status_scenario("delivered")
        sop = {"policy": "If the order is delivered, ask customer to confirm receipt."}
        turns = [
            {"turn_id": 1, "speaker": "AGENT", "content": "Your order is delivered."},
        ]
        sel = self._select(turns, scenario, sop=sop)
        self.assertEqual(len(sel.eligible), 0)
        codes = {e.reason_code for e in sel.exclusions}
        self.assertTrue("EXCLUDED_JUDGE_OVERLAP" in codes or "CONDITIONAL_OR_GENERIC_CLAIM" in codes)

        # validation-time enforcement for status substitution when SOP branches on trusted status
        evidence = factual.filter_supported_evidence(
            factual.build_available_evidence(scenario), scenario
        )
        ev_item = next(e for e in evidence if e["location"] == "scenario.order.shipping_status")
        source_turns = factual.extract_conversation_turns(turns)
        ctx = factual.build_factual_validation_context(
            conversation_id=1,
            source_conversation=source_turns,
            scenario=scenario,
            sop=sop,
            judge_definition=self.judges["factual_hallucination"],
            available_evidence=evidence,
        )
        payload = {
            "injection_metadata": {
                "target_judge": "factual_hallucination",
                "already_defective": False,
                "injection_status": "injected",
                "injection_turn_id": 1,
                "scope": "single_turn",
                "error_category": "order_status",
                "error_classification": "CONTRADICTED",
                "original_agent_claim": "Your order is delivered",
                "injected_claim": "Your order is shipped",
                "evidence_source_type": ev_item["source_type"],
                "evidence_source_location": ev_item["location"],
                "evidence_text": ev_item["text"],
                "contradiction_explanation": "x",
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
                    "notes": "",
                },
                "expected_target_verdict": "FAIL",
            },
            "injected_conversation": [{"turn_id": 1, "speaker": "AGENT", "content": "Your order is shipped."}],
        }
        with self.assertRaises(factual.FactualValidationError):
            factual.validate_factual_golden_record(payload, {}, ctx)

    def test_replacement_preserving_sop_branch_and_contradicting_evidence_can_remain_eligible(self):
        scenario = _status_scenario("delivered")
        sop = {"policy": "Handle shipping inquiries according to account verification rules."}
        turns = [{"turn_id": 1, "speaker": "AGENT", "content": "Your order is delivered."}]
        sel = self._select(turns, scenario, sop=sop)
        self.assertGreaterEqual(len(sel.eligible), 1)
        # validation with SOP that does not branch on status should allow contradiction
        evidence = factual.filter_supported_evidence(
            factual.build_available_evidence(scenario), scenario
        )
        ev_item = next(e for e in evidence if e["location"] == "scenario.order.shipping_status")
        source_turns = factual.extract_conversation_turns(turns)
        ctx = factual.build_factual_validation_context(
            conversation_id=1,
            source_conversation=source_turns,
            scenario=scenario,
            sop=sop,
            judge_definition=self.judges["factual_hallucination"],
            available_evidence=evidence,
        )
        payload = {
            "injection_metadata": {
                "target_judge": "factual_hallucination",
                "already_defective": False,
                "injection_status": "injected",
                "injection_turn_id": 1,
                "scope": "single_turn",
                "error_category": "order_status",
                "error_classification": "CONTRADICTED",
                "original_agent_claim": "Your order is delivered",
                "injected_claim": "Your order is shipped",
                "evidence_source_type": ev_item["source_type"],
                "evidence_source_location": ev_item["location"],
                "evidence_text": ev_item["text"],
                "contradiction_explanation": "x",
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
                    "notes": "",
                },
                "expected_target_verdict": "FAIL",
            },
            "injected_conversation": [{"turn_id": 1, "speaker": "AGENT", "content": "Your order is shipped."}],
        }
        rec = factual.validate_factual_golden_record(payload, {}, ctx)
        self.assertEqual(rec.injection_metadata.error_classification, "CONTRADICTED")

    def test_pre_existing_incoherent_sentence_rejected_or_flagged(self):
        scenario = _status_scenario("delivered")
        turns = [
            {"turn_id": 1, "speaker": "AGENT", "content": "Your order is delivered and shipped."},
        ]
        sel = self._select(turns, scenario, sop=None)
        self.assertEqual(len(sel.eligible), 0)
        codes = {e.reason_code for e in sel.exclusions}
        self.assertIn("PRE_EXISTING_INCOHERENT_SENTENCE", codes)

    def test_case_insensitive_evidence_matching_without_changing_original_text(self):
        scenario = {"personal": {"customer_name": "joyce wu"}, "flow": "a", "subflow": "b"}
        turns = [{"turn_id": 1, "speaker": "AGENT", "content": "I have found your account JOYCE WU."}]
        # _name_candidates requires capitalized name tokens in text; use mixed/capitalized form but different case than evidence
        turns = [{"turn_id": 1, "speaker": "AGENT", "content": "I have found your account Joyce Wu."}]
        sel = self._select(turns, scenario, sop=None)
        self.assertGreaterEqual(len(sel.eligible), 1)
        cand = sel.eligible[0]
        self.assertIn("Joyce Wu", cand.original_agent_claim)
        self.assertEqual(cand.evidence["text"], "joyce wu")  # preserved exact evidence text
        # ensure turn content unchanged by selection (selection doesn't mutate)
        self.assertEqual(turns[0]["content"], "I have found your account Joyce Wu.")

    def test_existing_non_factual_preparation_behavior_unchanged(self):
        # data_judges batch/template path and record shape should still work as before
        template = PROJECT_ROOT / "prompts/data_judges_injection/data_judges_injection_prompt_template.txt"
        sample = json.loads((PROJECT_ROOT / "data/source/abcd/data/abcd_sample.json").read_text(encoding="utf-8"))
        records = prepare_records(
            conversations=sample[:1],
            guidelines=self.guidelines,
            judges=self.judges,
            requested_judges=["missing_data"],
            conversation_count=1,
            rng=random.Random(0),
            prompt_template_path=template,
        )
        # may be 0 if schema mismatch in sample indexing; try any of first few
        if not records:
            records = prepare_records(
                conversations=sample[:3],
                guidelines=self.guidelines,
                judges=self.judges,
                requested_judges=["missing_data"],
                conversation_count=1,
                rng=random.Random(0),
                prompt_template_path=template,
            )
        self.assertTrue(records, "expected at least one non-factual prepared record from sample")
        rec = records[0]
        self.assertEqual(rec["target_judge"], "missing_data")
        self.assertIn("assembled_prompt", rec)
        self.assertIn("sampled_params", rec)
        self.assertNotIn("source_conversation", rec)  # factual-only context fields not added to non-factual
        self.assertIn("injection_mechanism", rec["sampled_params"])

        # existing generation-side schema validation for general outputs unchanged
        valid_general = {
            "injection_metadata": {
                "target_judge": "intent_understanding",
                "already_defective": False,
                "scenario_modified": False,
                "modified_scenario_fields": {},
                "injected_turn_indices": [2],
                "total_injected_errors": 1,
                "target_dialogue_phase": "early",
                "scope": "single_turn",
            },
            "injected_conversation": [
                {"turn_id": 1, "speaker": "CUSTOMER", "content": "hi"},
                {"turn_id": 2, "speaker": "AGENT", "content": "wrong intent"},
            ],
            "judge_evaluations": {
                "intent_understanding": {
                    "verdict": "FAIL",
                    "violations": [{"turn_id": 2, "speaker": "AGENT", "explanation": "x"}],
                    "overall_reasoning": "x",
                }
            },
        }
        out = GoldenRecordOutput.model_validate(valid_general)
        self.assertEqual(out.injection_metadata.target_judge, "intent_understanding")


if __name__ == "__main__":
    unittest.main()
