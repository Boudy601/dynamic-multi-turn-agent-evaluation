import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "dataset_generation"))

import factual_hallucination as factual  # noqa: E402
from generate_golden_dataset import build_factual_output_record  # noqa: E402
from prepare_prompts import (  # noqa: E402
    flatten_guidelines,
    load_judges,
    prepare_records,
)
from report_factual_quality import (  # noqa: E402
    build_quality_report,
    render_markdown,
)
from screen_prompts import (  # noqa: E402
    build_screening_request,
    screen_prepared_record,
    screening_error_outcome,
)

SCENARIO = {
    "personal": {"customer_name": "Joyce Wu"},
    "order": {"order_id": "2621786777", "shipping_status": "delivered"},
    "flow": "Shipping Issue",
    "subflow": "Shipping Status",
}

TURNS = [
    {"turn_id": 1, "speaker": "AGENT", "content": "Hello, how can I help you today?"},
    {"turn_id": 2, "speaker": "CUSTOMER", "content": "Where is my order?"},
    {"turn_id": 3, "speaker": "AGENT", "content": "Your order is delivered."},
    {"turn_id": 4, "speaker": "CUSTOMER", "content": "Thanks."},
    {"turn_id": 5, "speaker": "AGENT", "content": "Is there anything else I can help with?"},
]

ORIGINAL_CLAIM = "Your order is delivered"
INJECTED_CLAIM = "Your order is shipped"


def _conversation():
    return {
        "convo_id": 990001,
        "scenario": dict(SCENARIO),
        "original": [[turn["speaker"], turn["content"]] for turn in TURNS],
    }


def _judges():
    path = PROJECT_ROOT / "config" / "judges_definitions.json"
    return load_judges(json.loads(path.read_text(encoding="utf-8")), path)


def _guidelines():
    path = PROJECT_ROOT / "data/source/abcd/data/guidelines.json"
    return flatten_guidelines(json.loads(path.read_text(encoding="utf-8")), path)


def _prepared_record():
    return prepare_records(
        conversations=[_conversation()],
        guidelines=_guidelines(),
        judges=_judges(),
        requested_judges=["factual_hallucination"],
        conversation_count=1,
        rng=random.Random(0),
        prompt_template_path=(
            PROJECT_ROOT / "prompts/factual_hallucination_injection_prompt_template.txt"
        ),
    )[0]


def _selection():
    evidence = factual.filter_supported_evidence(
        factual.build_available_evidence(SCENARIO), SCENARIO
    )
    turns = factual.extract_conversation_turns(_conversation())
    return factual.select_factual_candidates(turns, SCENARIO, evidence)


def _screening_response(
    *, decision="ELIGIBLE", claim=ORIGINAL_CLAIM, evidence="delivered", reason="ok"
):
    return json.dumps(
        {
            "decision": decision,
            "target_claim": claim,
            "evidence_location": "scenario.order.shipping_status",
            "evidence_text": evidence,
            "reason": reason,
        }
    )


def _validation_context():
    turns = factual.extract_conversation_turns(_conversation())
    evidence = factual.filter_supported_evidence(
        factual.build_available_evidence(SCENARIO), SCENARIO
    )
    return factual.build_factual_validation_context(
        conversation_id=990001,
        source_conversation=turns,
        scenario=SCENARIO,
        sop={},
        judge_definition=_judges()["factual_hallucination"],
        available_evidence=evidence,
    )


def _payload(
    *,
    original_claim=ORIGINAL_CLAIM,
    injected_claim=INJECTED_CLAIM,
    injection_turn_id=3,
    scenario_modified=False,
    extra_edit_turn=None,
):
    turns = [dict(turn) for turn in factual.extract_conversation_turns(_conversation())]
    target = next(turn for turn in turns if turn["turn_id"] == injection_turn_id)
    target["content"] = target["content"].replace(original_claim, injected_claim, 1)
    if extra_edit_turn is not None:
        extra = next(turn for turn in turns if turn["turn_id"] == extra_edit_turn)
        extra["content"] = extra["content"] + " Extra."
    return {
        "injection_metadata": {
            "target_judge": "factual_hallucination",
            "already_defective": False,
            "injection_status": "injected",
            "injection_turn_id": injection_turn_id,
            "scope": "single_turn",
            "error_category": "order_status",
            "error_classification": "CONTRADICTED",
            "original_agent_claim": original_claim,
            "injected_claim": injected_claim,
            "evidence_source_type": "scenario",
            "evidence_source_location": "scenario.order.shipping_status",
            "evidence_text": "delivered",
            "contradiction_explanation": "The Agent changed a delivered order to shipped.",
            "scenario_modified": scenario_modified,
            "modified_scenario_fields": {"order.shipping_status": "shipped"}
            if scenario_modified
            else {},
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
        "injected_conversation": turns,
    }


class PreGenerationScreeningTests(unittest.TestCase):
    def test_python_candidate_requires_llm_confirmation(self):
        stage = factual.python_screening_stage(_selection())
        self.assertEqual(stage.decision, "REVIEW")
        self.assertTrue(stage.requires_llm)
        self.assertEqual(stage.target_claim, ORIGINAL_CLAIM)

    def test_screening_not_run_never_returns_eligible(self):
        result = factual.screen_factual_candidates(_selection(), None)
        self.assertEqual(result.decision, "REVIEW")
        self.assertIn("not run", result.reason)

    def test_llm_screening_eligible_requires_exact_echo(self):
        stage = factual.python_screening_stage(_selection())
        result = factual.apply_llm_screening(
            stage, _screening_response(), _selection()
        )
        self.assertEqual(result.decision, "ELIGIBLE")
        self.assertEqual(result.target_claim, ORIGINAL_CLAIM)
        self.assertEqual(result.evidence_text, "delivered")

    def test_llm_screening_non_matching_echo_goes_to_review(self):
        stage = factual.python_screening_stage(_selection())
        result = factual.apply_llm_screening(
            stage,
            _screening_response(claim="A different claim"),
            _selection(),
        )
        self.assertEqual(result.decision, "REVIEW")
        self.assertIn("did not echo", result.reason)

    def test_llm_screening_reject_and_review(self):
        stage = factual.python_screening_stage(_selection())
        rejected = factual.apply_llm_screening(
            stage,
            _screening_response(decision="REJECT", reason="not a fact"),
            _selection(),
        )
        self.assertEqual(rejected.decision, "REJECT")
        self.assertEqual(rejected.reason, "not a fact")

        reviewed = factual.apply_llm_screening(
            stage,
            _screening_response(decision="REVIEW", reason="evidence unclear"),
            _selection(),
        )
        self.assertEqual(reviewed.decision, "REVIEW")
        self.assertEqual(reviewed.reason, "evidence unclear")

    def test_malformed_screening_response_goes_to_review(self):
        stage = factual.python_screening_stage(_selection())
        result = factual.apply_llm_screening(stage, "not json", _selection())
        self.assertEqual(result.decision, "REVIEW")
        self.assertIn("not valid JSON", result.reason)

    def test_prepared_record_can_be_screened_and_unlocks_generation(self):
        record = _prepared_record()
        self.assertEqual(record["screening"]["decision"], "REVIEW")
        self.assertTrue(record["screening"]["requires_llm"])
        self.assertIn("factual_selection", record)
        self.assertNotIn("SCREENING DECISION: ELIGIBLE", record["assembled_prompt"])

        messages = build_screening_request(record)
        self.assertIsNotNone(messages)
        self.assertIn("Screen one factual-hallucination candidate", messages[0]["content"])

        outcome = screen_prepared_record(record, _screening_response())
        self.assertEqual(outcome.decision, "ELIGIBLE")
        screened = outcome.record
        self.assertIsNotNone(screened)
        self.assertEqual(
            screened["screened_candidate"]["target_claim"], ORIGINAL_CLAIM
        )
        self.assertIn(
            "SCREENING DECISION: ELIGIBLE", screened["assembled_prompt"]
        )

        blocked = factual.evaluate_generation_gate(record)
        self.assertFalse(blocked.ready)
        self.assertEqual(blocked.decision, "REVIEW")

        gate = factual.evaluate_generation_gate(screened)
        self.assertTrue(gate.ready)
        self.assertEqual(gate.decision, "ELIGIBLE")

    def test_unscreened_record_is_always_blocked(self):
        record = _prepared_record()
        legacy = {key: value for key, value in record.items() if key != "screening"}
        for allow_legacy in (False, True):
            with self.subTest(allow_legacy=allow_legacy):
                blocked = factual.evaluate_generation_gate(
                    legacy, allow_unscreened=allow_legacy
                )
                self.assertFalse(blocked.ready)
                self.assertEqual(blocked.decision, "REVIEW")
                self.assertIn("screen_prompts.py", blocked.reason)

    def test_fabricated_eligible_without_screened_candidate_is_blocked(self):
        record = _prepared_record()
        record["screening"] = {"decision": "ELIGIBLE", "reason": "self-reported"}
        gate = factual.evaluate_generation_gate(record)
        self.assertFalse(gate.ready)
        self.assertIn("screened_candidate", gate.reason)

    def test_legacy_record_without_selection_is_rescreened_from_context(self):
        record = _prepared_record()
        legacy = {
            key: value
            for key, value in record.items()
            if key not in {"factual_selection", "screening", "screened_candidate"}
        }
        selection = factual.selection_from_prompt_record(legacy)
        self.assertEqual(len(selection.eligible), 1)
        self.assertIsNotNone(build_screening_request(legacy))
        outcome = screen_prepared_record(legacy, _screening_response())
        self.assertEqual(outcome.decision, "ELIGIBLE")

    def test_screening_provider_error_fails_closed(self):
        record = _prepared_record()
        outcome = screening_error_outcome(record, RuntimeError("timeout"))
        self.assertEqual(outcome.decision, "REVIEW")
        self.assertIsNone(outcome.record)
        self.assertIn("must not be generated", outcome.audit["reason"])


class EvidenceVerificationTests(unittest.TestCase):
    def _evidence(self):
        return next(
            item
            for item in factual.filter_supported_evidence(
                factual.build_available_evidence(SCENARIO), SCENARIO
            )
            if item["location"] == "scenario.order.shipping_status"
        )

    def test_direct_contradiction_is_contradicted(self):
        result = factual.verify_factual_evidence(
            original_claim=ORIGINAL_CLAIM,
            injected_claim=INJECTED_CLAIM,
            evidence=self._evidence(),
            scenario=SCENARIO,
            error_category="order_status",
        )
        self.assertEqual(result.classification, "CONTRADICTED")
        self.assertTrue(result.entity_match)
        self.assertTrue(result.attribute_match)
        self.assertTrue(result.context_match)

    def test_same_value_is_supported_not_contradicted(self):
        result = factual.verify_factual_evidence(
            original_claim=ORIGINAL_CLAIM,
            injected_claim=ORIGINAL_CLAIM,
            evidence=self._evidence(),
            scenario=SCENARIO,
            error_category="order_status",
        )
        self.assertEqual(result.classification, "SUPPORTED")

    def test_vague_restatement_is_unsupported(self):
        result = factual.verify_factual_evidence(
            original_claim=ORIGINAL_CLAIM,
            injected_claim="Your order is being handled",
            evidence=self._evidence(),
            scenario=SCENARIO,
            error_category="order_status",
        )
        self.assertEqual(result.classification, "UNSUPPORTED")

    def test_mismatched_attribute_is_not_verifiable(self):
        result = factual.verify_factual_evidence(
            original_claim=ORIGINAL_CLAIM,
            injected_claim=INJECTED_CLAIM,
            evidence=self._evidence(),
            scenario=SCENARIO,
            error_category="customer_identity",
        )
        self.assertEqual(result.classification, "NOT_VERIFIABLE")
        self.assertFalse(result.attribute_match)

    def test_unsupported_evidence_location_is_not_verifiable(self):
        result = factual.verify_factual_evidence(
            original_claim=ORIGINAL_CLAIM,
            injected_claim=INJECTED_CLAIM,
            evidence={
                "evidence_id": "scenario.unknown.value",
                "source_type": "scenario",
                "location": "scenario.unknown.value",
                "text": "whatever",
            },
            scenario=SCENARIO,
            error_category="order_status",
        )
        self.assertEqual(result.classification, "NOT_VERIFIABLE")

    def test_email_local_part_substitution_is_contradicted(self):
        scenario = {"personal": {"email": "cm131791@email.com"}}
        evidence = {
            "evidence_id": "scenario.personal.email",
            "source_type": "scenario",
            "location": "scenario.personal.email",
            "text": "cm131791@email.com",
        }
        result = factual.verify_factual_evidence(
            original_claim="the code has been sent to cm131791@email.com",
            injected_claim="the code has been sent to cm131792@email.com",
            evidence=evidence,
            scenario=scenario,
            error_category="customer_identity",
            all_evidence=[evidence],
        )
        self.assertEqual(result.classification, "CONTRADICTED")
        self.assertIn("directly contradicts", result.reason)

    def test_email_domain_substitution_is_contradicted(self):
        scenario = {"personal": {"email": "cm131791@email.com"}}
        evidence = {
            "evidence_id": "scenario.personal.email",
            "source_type": "scenario",
            "location": "scenario.personal.email",
            "text": "cm131791@email.com",
        }
        result = factual.verify_factual_evidence(
            original_claim="the code has been sent to cm131791@email.com",
            injected_claim="the code has been sent to cm131791@mail.com",
            evidence=evidence,
            scenario=scenario,
            error_category="customer_identity",
            all_evidence=[evidence],
        )
        self.assertEqual(result.classification, "CONTRADICTED")

    def test_phone_substitution_is_contradicted(self):
        scenario = {"personal": {"phone": "(186) 372-1256"}}
        evidence = {
            "evidence_id": "scenario.personal.phone",
            "source_type": "scenario",
            "location": "scenario.personal.phone",
            "text": "(186) 372-1256",
        }
        result = factual.verify_factual_evidence(
            original_claim="I can call you at (186) 372-1256",
            injected_claim="I can call you at (186) 372-1257",
            evidence=evidence,
            scenario=scenario,
            error_category="customer_identity",
            all_evidence=[evidence],
        )
        self.assertEqual(result.classification, "CONTRADICTED")

    def test_email_substitution_still_rejects_unrelated_identifier_change(self):
        scenario = {"personal": {"email": "cm131791@email.com"}}
        evidence = {
            "evidence_id": "scenario.personal.email",
            "source_type": "scenario",
            "location": "scenario.personal.email",
            "text": "cm131791@email.com",
        }
        result = factual.verify_factual_evidence(
            original_claim="email cm131791@email.com for order 12345",
            injected_claim="email cm131792@email.com for order 54321",
            evidence=evidence,
            scenario=scenario,
            error_category="customer_identity",
            all_evidence=[evidence],
        )
        self.assertEqual(result.classification, "NOT_VERIFIABLE")
        self.assertIn("identifier", result.reason)

    def test_numeric_replacement_colliding_with_other_trusted_value_is_rejected(self):
        scenario = {"product": {"amounts": [64, 94]}}
        evidence_items = [
            {
                "evidence_id": "scenario.product.amounts[0]",
                "source_type": "scenario",
                "location": "scenario.product.amounts[0]",
                "text": "64",
            },
            {
                "evidence_id": "scenario.product.amounts[1]",
                "source_type": "scenario",
                "location": "scenario.product.amounts[1]",
                "text": "94",
            },
        ]
        result = factual.verify_factual_evidence(
            original_claim="Your refund is $64",
            injected_claim="Your refund is $94",
            evidence=evidence_items[0],
            scenario=scenario,
            error_category="order_amount",
            all_evidence=evidence_items,
        )
        self.assertEqual(result.classification, "NOT_VERIFIABLE")
        self.assertIn("collides", result.reason)
        self.assertIn("amounts[1]", result.reason)

    def test_negative_numeric_replacement_is_rejected(self):
        scenario = {"product": {"amounts": [64]}}
        evidence = {
            "evidence_id": "scenario.product.amounts[0]",
            "source_type": "scenario",
            "location": "scenario.product.amounts[0]",
            "text": "64",
        }
        result = factual.verify_factual_evidence(
            original_claim="Your refund is $64",
            injected_claim="Your refund is $-94",
            evidence=evidence,
            scenario=scenario,
            error_category="order_amount",
            all_evidence=[evidence],
        )
        self.assertEqual(result.classification, "NOT_VERIFIABLE")
        self.assertIn("negative", result.reason)

    def test_malformed_email_replacement_is_rejected(self):
        scenario = {"personal": {"email": "cm131791@email.com"}}
        evidence = {
            "evidence_id": "scenario.personal.email",
            "source_type": "scenario",
            "location": "scenario.personal.email",
            "text": "cm131791@email.com",
        }
        result = factual.verify_factual_evidence(
            original_claim="the code has been sent to cm131791@email.com",
            injected_claim="the code has been sent to cm131792@email..com",
            evidence=evidence,
            scenario=scenario,
            error_category="customer_identity",
            all_evidence=[evidence],
        )
        self.assertEqual(result.classification, "NOT_VERIFIABLE")
        self.assertIn("not a valid address", result.reason)


class ControlledInjectionValidationTests(unittest.TestCase):
    def test_valid_single_injection_preserves_conversation_and_schema(self):
        context = _validation_context()
        record = factual.validate_factual_golden_record(_payload(), {}, context)
        output = build_factual_output_record(990001, record, context)
        self.assertEqual(
            set(output),
            {
                "conversation_id",
                "injection_metadata",
                "injected_conversation",
                "judge_evaluations",
                "audit_context",
                "validation_report",
            },
        )
        modified = output["injected_conversation"]
        source = factual.extract_conversation_turns(_conversation())
        changed = [
            turn["turn_id"]
            for turn, original in zip(modified, source)
            if turn["content"] != original["content"]
        ]
        self.assertEqual(changed, [3])
        self.assertEqual(modified[2]["content"], "Your order is shipped.")
        self.assertEqual(output["injection_metadata"]["scenario_modified"], False)
        self.assertEqual(
            output["injection_metadata"]["modified_scenario_fields"], {}
        )
        self.assertEqual(
            record.validation_report.automated_checks.evidence_classification,
            "CONTRADICTED",
        )
        self.assertTrue(record.validation_report.automated_checks.minimal_perturbation)

    def test_golden_output_preserves_audit_context_and_validation_report(self):
        context = _validation_context()
        record = factual.validate_factual_golden_record(_payload(), {}, context)
        output = build_factual_output_record(990001, record, context)

        audit = output["audit_context"]
        self.assertEqual(audit["source_conversation"], context["source_conversation"])
        self.assertEqual(audit["source_scenario"], context["source_scenario"])
        self.assertEqual(audit["source_sop"], context["source_sop"])
        self.assertEqual(
            audit["available_evidence"], context["available_evidence"]
        )
        self.assertEqual(
            audit["judge_definition"], context["judge_definition"]
        )

        report = output["validation_report"]
        self.assertEqual(
            report["evidence_verification"]["classification"], "CONTRADICTED"
        )
        self.assertTrue(report["automated_checks"]["exactly_one_injected_claim"])
        self.assertTrue(report["automated_checks"]["minimal_perturbation"])
        self.assertEqual(
            output["injection_metadata"]["original_agent_claim"], ORIGINAL_CLAIM
        )

    def test_output_recovers_source_conversation_without_input_record(self):
        context = _validation_context()
        record = factual.validate_factual_golden_record(_payload(), {}, context)
        output = build_factual_output_record(990001, record)
        self.assertEqual(
            output["audit_context"]["source_conversation"],
            context["source_conversation"],
        )
        self.assertEqual(
            output["audit_context"]["source_scenario"], context["source_scenario"]
        )
        self.assertEqual(
            output["audit_context"]["available_evidence"],
            context["available_evidence"],
        )

    def test_customer_turn_edit_is_rejected(self):
        with self.assertRaisesRegex(
            factual.FactualValidationError, "Only the single target AGENT turn"
        ):
            factual.validate_factual_golden_record(
                _payload(extra_edit_turn=2), {}, _validation_context()
            )

    def test_second_agent_edit_is_rejected(self):
        with self.assertRaisesRegex(
            factual.FactualValidationError, "Only the single target AGENT turn"
        ):
            factual.validate_factual_golden_record(
                _payload(extra_edit_turn=5), {}, _validation_context()
            )

    def test_scenario_modification_is_rejected(self):
        with self.assertRaisesRegex(
            factual.FactualValidationError, "scenario_modified"
        ):
            factual.validate_factual_golden_record(
                _payload(scenario_modified=True), {}, _validation_context()
            )

    def test_procedural_side_effect_is_rejected(self):
        with self.assertRaisesRegex(
            factual.FactualValidationError, "procedural action"
        ):
            factual.validate_factual_golden_record(
                _payload(
                    injected_claim="Your order is shipped and I will issue a refund"
                ),
                {},
                _validation_context(),
            )

    def test_incoherent_injected_claim_is_rejected(self):
        with self.assertRaisesRegex(
            factual.FactualValidationError, "incoherent"
        ):
            factual.validate_factual_golden_record(
                _payload(injected_claim="Your order is shipped and delivered"),
                {},
                _validation_context(),
            )

    def test_missing_input_context_is_rejected(self):
        with self.assertRaisesRegex(
            factual.FactualValidationError, "requires the prepared source context"
        ):
            factual.validate_factual_golden_record(_payload())


class PostGenerationDispositionTests(unittest.TestCase):
    def test_ambiguous_validation_goes_to_review(self):
        result = factual.classify_post_generation_failure(
            "Evidence 'x' is not a recognized single shipping status"
        )
        self.assertEqual(result.disposition, "REVIEW")

    def test_structural_validation_is_rejected(self):
        result = factual.classify_post_generation_failure(
            "Only the single target AGENT turn may change"
        )
        self.assertEqual(result.disposition, "REJECT")
        self.assertEqual(result.reason_code, "POST_GENERATION_REJECTED")

    def test_review_record_shape(self):
        review = factual.FactualReviewRecord(
            conversation_id=1,
            target_judge="factual_hallucination",
            review_status="REVIEW",
            stage="post_generation",
            reason="ambiguous",
        )
        self.assertEqual(review.model_dump(mode="json")["stage"], "post_generation")


class QualityMetricsTests(unittest.TestCase):
    def _accepted_row(self):
        return {
            "conversation_id": 1,
            "injection_metadata": {
                "injection_status": "injected",
                "target_judge": "factual_hallucination",
            },
        }

    def test_run_metrics_cover_screening_reviews_and_validation_rate(self):
        metrics = factual.compute_run_metrics(
            accepted_rows=[self._accepted_row()],
            rejections=[{"conversation_id": 2, "rejection_code": "OTHER"}],
            failures=[
                {
                    "conversation_id": 3,
                    "validation_errors": ["Only the single target AGENT turn may change"],
                }
            ],
            reviews=[{"conversation_id": 4, "stage": "screening"}],
            screening_records=[
                {"conversation_id": 1, "decision": "ELIGIBLE"},
                {"conversation_id": 2, "decision": "REJECT"},
                {"conversation_id": 4, "decision": "REVIEW"},
            ],
        )
        self.assertEqual(metrics.injected_sample_count, 1)
        self.assertEqual(metrics.rejection_count, 1)
        self.assertEqual(metrics.failed_generation_count, 1)
        self.assertEqual(metrics.reviewed_sample_count, 1)
        self.assertEqual(metrics.screening_eligible_count, 1)
        self.assertEqual(metrics.screening_reject_count, 1)
        self.assertEqual(metrics.screening_review_count, 1)
        self.assertEqual(metrics.validation_attempt_count, 2)
        self.assertEqual(metrics.validation_success_count, 1)
        self.assertEqual(metrics.validation_success_rate, 0.5)
        self.assertEqual(metrics.total_input_records, 6)
        self.assertEqual(
            metrics.failure_reasons, {"CONVERSATION_STRUCTURE_MUTATED": 1}
        )

    def test_missing_failures_leave_failure_rate_none(self):
        metrics = factual.compute_run_metrics(accepted_rows=[self._accepted_row()])
        self.assertIsNone(metrics.failed_generation_count)
        self.assertIsNone(metrics.screening_eligible_count)
        self.assertIsNone(metrics.validation_attempt_count)
        self.assertIsNone(metrics.validation_success_rate)

    def test_legacy_compute_quality_metrics_still_works(self):
        metrics = factual.compute_quality_metrics([])
        self.assertIsNone(metrics.validation_success_rate)
        self.assertIsNone(metrics.screening_eligible_count)

    def test_report_reads_files_and_renders_markdown(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def write(name, rows):
                path = root / name
                path.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )
                return path

            golden = write("golden.jsonl", [self._accepted_row()])
            failed = write(
                "failed.jsonl",
                [{"conversation_id": 3, "validation_errors": ["json decode"]}],
            )
            rejected = write(
                "rejected.jsonl", [{"conversation_id": 2, "rejection_code": "OTHER"}]
            )
            review = write(
                "review.jsonl", [{"conversation_id": 4, "stage": "screening"}]
            )
            screening = write(
                "screening.jsonl",
                [
                    {"conversation_id": 1, "decision": "ELIGIBLE"},
                    {"conversation_id": 2, "decision": "REJECT"},
                    {"conversation_id": 4, "decision": "REVIEW"},
                ],
            )
            report = build_quality_report(
                golden_path=golden,
                failed_path=failed,
                rejected_path=rejected,
                review_path=review,
                screening_path=screening,
            )
            self.assertEqual(report["counts"]["accepted"], 1)
            self.assertEqual(report["counts"]["failed"], 1)
            self.assertEqual(report["counts"]["reviewed"], 1)
            self.assertEqual(report["counts"]["screening"]["ELIGIBLE"], 1)
            self.assertEqual(report["metrics"]["validation_success_rate"], 0.5)
            markdown = render_markdown(report)
            self.assertIn("Validation success rate: 50.0%", markdown)
            self.assertIn("Factual Hallucination (L5) Quality Report", markdown)


class BackwardCompatibilityTests(unittest.TestCase):
    def test_prepared_record_keeps_existing_fields(self):
        record = _prepared_record()
        for key in (
            "conversation_id",
            "target_judge",
            "sampled_params",
            "assembled_prompt",
            "source_conversation",
            "source_scenario",
            "source_sop",
            "available_evidence",
            "judge_definition",
            "human_review_flags",
        ):
            self.assertIn(key, record)
        self.assertIn("eligible candidate", record["assembled_prompt"].lower())
        self.assertEqual(record["target_judge"], "factual_hallucination")

    def test_old_golden_row_core_keys_remain(self):
        context = _validation_context()
        record = factual.validate_factual_golden_record(_payload(), {}, context)
        output = build_factual_output_record(990001, record, context)
        core_keys = {
            "conversation_id",
            "injection_metadata",
            "injected_conversation",
            "judge_evaluations",
        }
        self.assertTrue(core_keys.issubset(output))
        self.assertIn("audit_context", output)
        self.assertIn("validation_report", output)
        self.assertEqual(output["injection_metadata"]["target_judge"], "factual_hallucination")
        self.assertEqual(
            output["judge_evaluations"]["factual_hallucination"]["verdict"], "FAIL"
        )

    def test_selection_remains_deterministic_and_non_mutating(self):
        turns = factual.extract_conversation_turns(_conversation())
        before = json.dumps(turns, sort_keys=True)
        selection = factual.select_factual_candidates(
            turns,
            SCENARIO,
            factual.filter_supported_evidence(
                factual.build_available_evidence(SCENARIO), SCENARIO
            ),
        )
        self.assertEqual(len(selection.eligible), 1)
        self.assertEqual(json.dumps(turns, sort_keys=True), before)


class GenerationPipelineGateTests(unittest.TestCase):
    def test_default_generation_input_is_the_screened_file(self):
        import sys
        from unittest import mock

        import generate_golden_dataset as generate

        with mock.patch.object(sys, "argv", ["generate_golden_dataset.py"]):
            args = generate.parse_args()
        self.assertEqual(
            Path(args.input).name, "prepared_prompts_screened.jsonl"
        )

    def test_review_rows_are_not_treated_as_completed_on_resume(self):
        import generate_golden_dataset as generate

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.jsonl"
            output_path = root / "output.jsonl"
            rejected_path = root / "rejected.jsonl"
            review_path = root / "review.jsonl"

            def write(path, rows):
                path.write_text(
                    "\n".join(json.dumps(row) for row in rows) + "\n",
                    encoding="utf-8",
                )

            write(input_path, [{"conversation_id": 1}, {"conversation_id": 2}, {"conversation_id": 3}])
            write(output_path, [{"conversation_id": 1}])
            write(rejected_path, [{"conversation_id": 2}])
            write(review_path, [{"conversation_id": 3, "review_status": "REVIEW"}])

            pending = generate.load_pending_records(
                input_path, output_path, rejected_path
            )
            self.assertEqual(
                [row["conversation_id"] for row in pending], [3]
            )

    def test_process_record_blocks_unscreened_record_without_model_call(self):
        import asyncio
        from unittest import mock

        import generate_golden_dataset as generate

        record = _prepared_record()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                "output": root / "out.jsonl",
                "failed": root / "failed.jsonl",
                "rejected": root / "rejected.jsonl",
                "review": root / "review.jsonl",
            }
            locks = {name: asyncio.Lock() for name in paths}
            with mock.patch.object(
                generate,
                "request_with_backoff",
                side_effect=AssertionError("model must not be called"),
            ):
                asyncio.run(
                    generate.process_record(
                        record,
                        object(),
                        "gemini",
                        "test-model",
                        asyncio.Semaphore(1),
                        paths["output"],
                        paths["failed"],
                        paths["rejected"],
                        locks["output"],
                        locks["failed"],
                        locks["rejected"],
                        review_path=paths["review"],
                        review_lock=locks["review"],
                        allow_unscreened_factual=True,
                    )
                )
            review_rows = [
                json.loads(line)
                for line in paths["review"].read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(review_rows[0]["review_status"], "REVIEW")
            self.assertEqual(review_rows[0]["stage"], "screening")
            self.assertFalse(paths["output"].exists())
            self.assertFalse(paths["failed"].exists())

    def test_process_record_accepts_valid_screened_generation(self):
        import asyncio
        from unittest import mock

        import generate_golden_dataset as generate

        record = screen_prepared_record(
            _prepared_record(), _screening_response()
        ).record

        async def fake_request(*args, **kwargs):
            return json.dumps(_payload())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                "output": root / "out.jsonl",
                "failed": root / "failed.jsonl",
                "rejected": root / "rejected.jsonl",
                "review": root / "review.jsonl",
            }
            locks = {name: asyncio.Lock() for name in paths}
            with mock.patch.object(generate, "request_with_backoff", fake_request):
                asyncio.run(
                    generate.process_record(
                        record,
                        object(),
                        "gemini",
                        "test-model",
                        asyncio.Semaphore(1),
                        paths["output"],
                        paths["failed"],
                        paths["rejected"],
                        locks["output"],
                        locks["failed"],
                        locks["rejected"],
                        review_path=paths["review"],
                        review_lock=locks["review"],
                    )
                )
            output_rows = [
                json.loads(line)
                for line in paths["output"].read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                output_rows[0]["injection_metadata"]["candidate_id"],
                _selection().eligible[0].candidate_id,
            )
            self.assertEqual(
                output_rows[0]["injection_metadata"]["screening_decision"], "ELIGIBLE"
            )
            self.assertEqual(
                output_rows[0]["judge_evaluations"]["factual_hallucination"]["verdict"],
                "FAIL",
            )
            self.assertEqual(
                output_rows[0]["audit_context"]["source_conversation"],
                record["source_conversation"],
            )
            self.assertEqual(
                output_rows[0]["validation_report"]["evidence_verification"][
                    "classification"
                ],
                "CONTRADICTED",
            )
            self.assertFalse(paths["failed"].exists())
            self.assertFalse(paths["review"].exists())

    def test_process_record_writes_screening_reject_without_model_call(self):
        import asyncio
        from unittest import mock

        import generate_golden_dataset as generate

        record = _prepared_record()
        record["screening"] = {"decision": "REJECT", "reason": "not a fact"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                "output": root / "out.jsonl",
                "failed": root / "failed.jsonl",
                "rejected": root / "rejected.jsonl",
                "review": root / "review.jsonl",
            }
            locks = {name: asyncio.Lock() for name in paths}
            with mock.patch.object(
                generate,
                "request_with_backoff",
                side_effect=AssertionError("model must not be called"),
            ):
                asyncio.run(
                    generate.process_record(
                        record,
                        object(),
                        "gemini",
                        "test-model",
                        asyncio.Semaphore(1),
                        paths["output"],
                        paths["failed"],
                        paths["rejected"],
                        locks["output"],
                        locks["failed"],
                        locks["rejected"],
                        review_path=paths["review"],
                        review_lock=locks["review"],
                    )
                )
            rejected_rows = [
                json.loads(line)
                for line in paths["rejected"].read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(rejected_rows[0]["rejection_code"], "SCREENING_REJECTED")
            self.assertEqual(rejected_rows[0]["rejection_reason"], "not a fact")
            self.assertFalse(paths["output"].exists())
            self.assertFalse(paths["review"].exists())

    def test_process_record_routes_ambiguous_validation_to_review(self):
        import asyncio
        from unittest import mock

        import generate_golden_dataset as generate

        record = screen_prepared_record(
            _prepared_record(), _screening_response()
        ).record
        self.assertIsNotNone(record)
        ambiguous_payload = _payload(injected_claim="Your order is being handled")

        async def fake_request(*args, **kwargs):
            return json.dumps(ambiguous_payload)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                "output": root / "out.jsonl",
                "failed": root / "failed.jsonl",
                "rejected": root / "rejected.jsonl",
                "review": root / "review.jsonl",
            }
            locks = {name: asyncio.Lock() for name in paths}
            with mock.patch.object(generate, "request_with_backoff", fake_request):
                asyncio.run(
                    generate.process_record(
                        record,
                        object(),
                        "gemini",
                        "test-model",
                        asyncio.Semaphore(1),
                        paths["output"],
                        paths["failed"],
                        paths["rejected"],
                        locks["output"],
                        locks["failed"],
                        locks["rejected"],
                        review_path=paths["review"],
                        review_lock=locks["review"],
                    )
                )
            review_rows = [
                json.loads(line)
                for line in paths["review"].read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(review_rows[0]["stage"], "post_generation")
            self.assertIn("human review", review_rows[0]["reason"])
            self.assertEqual(review_rows[0]["target_claim"], ORIGINAL_CLAIM)
            self.assertFalse(paths["failed"].exists())


class ScreeningRunnerTests(unittest.TestCase):
    def test_screen_records_uses_provider_response_and_fails_closed(self):
        import asyncio
        from unittest import mock

        import screen_prompts

        eligible = _prepared_record()
        failing = dict(eligible)
        failing["conversation_id"] = 990002
        calls: list[int] = []

        async def fake_request(client, semaphore, platform, model, messages, timeout):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("network down")
            return _screening_response()

        async def run():
            return await screen_prompts.screen_records(
                [eligible, failing],
                client=object(),
                platform="gemini",
                model="test-model",
                semaphore=asyncio.Semaphore(1),
            )

        with mock.patch.object(
            screen_prompts, "request_with_backoff", fake_request
        ):
            outcomes = asyncio.run(run())
        self.assertEqual([outcome.decision for outcome in outcomes], ["ELIGIBLE", "REVIEW"])
        self.assertIsNone(outcomes[1].record)


if __name__ == "__main__":
    unittest.main()
