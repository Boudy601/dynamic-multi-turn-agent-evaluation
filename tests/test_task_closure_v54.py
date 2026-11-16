"""Structural regression tests for Task Closure V5.4 in the core generator.

Run from repository root:
  python -m unittest discover -s tests -p 'test_task_closure_v54.py' -v

Requires the dependencies listed in requirements.txt. These tests do not call an LLM.
"""
import asyncio
import copy
import json
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src' / 'dataset_generation'))
from generate_golden_dataset import validate_record, validate_task_closure_source, process_record  # noqa:E402


class V54SchemaTests(unittest.TestCase):
    def setUp(self):
        self.source = [
            {'turn_id': 1, 'speaker': 'CUSTOMER', 'content': 'Please send a code.'},
            {'turn_id': 2, 'speaker': 'AGENT', 'content': 'I will create the code.'},
            {'turn_id': 3, 'speaker': 'AGENT', 'content': 'Your code is SAVE20.'},
            {'turn_id': 4, 'speaker': 'CUSTOMER', 'content': 'Thanks for the code.'},
        ]
        self.data = {
            'injection_metadata': {
                'target_judge': 'task_closure',
                'already_defective': False,
                'scenario_modified': False,
                'modified_scenario_fields': {},
                'injected_turn_indices': [3],
                'customer_adjusted_turn_indices': [4],
                'total_injected_errors': 1,
                'target_dialogue_phase': 'late',
                'scope': 'single_turn',
            },
            'injected_conversation': copy.deepcopy(self.source),
            'judge_evaluations': {
                'task_closure': {
                    'verdict': 'FAIL',
                    'violations': [{'turn_id': 3, 'speaker': 'AGENT', 'explanation': 'Code never delivered.'}],
                    'overall_reasoning': 'The edited response withholds the code.',
                }
            },
        }
        self.data['injected_conversation'][2]['content'] = 'A code has been created.'
        self.data['injected_conversation'][3]['content'] = 'Thanks for the update.'
        self.params = {'target_error_count': 1, 'target_dialogue_phase': 'auto'}

    def check(self, data=None, source=None):
        record = validate_record(data or self.data, self.params, 'task_closure')
        validate_task_closure_source(record, source if source is not None else self.source)
        return record

    def assert_invalid(self, fragment, data=None, source=None):
        with self.assertRaises((ValueError, Exception)) as e:
            self.check(data, source)
        self.assertIn(fragment, str(e.exception))

    def test_v54_valid_downstream_customer_edit(self):
        rec = self.check()
        self.assertEqual(rec.injection_metadata.customer_adjusted_turn_indices, [4])

    def test_no_customer_edits_still_valid(self):
        d = copy.deepcopy(self.data)
        d['injection_metadata']['customer_adjusted_turn_indices'] = []
        d['injected_conversation'][3]['content'] = self.source[3]['content']
        self.check(d)

    def test_old_task_closure_without_customer_field_still_valid(self):
        d = copy.deepcopy(self.data)
        del d['injection_metadata']['customer_adjusted_turn_indices']
        d['injected_conversation'][3]['content'] = self.source[3]['content']
        rec = self.check(d)
        self.assertEqual(rec.injection_metadata.customer_adjusted_turn_indices, [])

    def test_undeclared_customer_edit_rejected(self):
        d = copy.deepcopy(self.data)
        d['injection_metadata']['customer_adjusted_turn_indices'] = []
        self.assert_invalid('changed CUSTOMER', d)

    def test_declared_unmodified_customer_rejected(self):
        d = copy.deepcopy(self.data)
        d['injected_conversation'][3]['content'] = self.source[3]['content']
        self.assert_invalid('changed CUSTOMER', d)

    def test_customer_edit_must_be_after_modified_agent(self):
        d = copy.deepcopy(self.data)
        d['injected_conversation'][0]['content'] = 'Please give me a code.'
        d['injected_conversation'][3]['content'] = self.source[3]['content']
        d['injection_metadata']['customer_adjusted_turn_indices'] = [1]
        self.assert_invalid('downstream', d)

    def test_action_immutable(self):
        d = copy.deepcopy(self.data)
        source = copy.deepcopy(self.source)
        source[1]['speaker'] = 'ACTION'
        source[1]['content'] = 'Created code'
        d['injected_conversation'][1]['speaker'] = 'ACTION'
        d['injected_conversation'][1]['content'] = 'Another code'
        self.assert_invalid('ACTION', d, source)

    def test_undeclared_agent_edit(self):
        d = copy.deepcopy(self.data)
        d['injected_conversation'][1]['content'] = 'Changed unrelated message'
        self.assert_invalid('changed AGENT', d)

    def test_duplicate_customer_id_rejected(self):
        d = copy.deepcopy(self.data)
        d['injection_metadata']['customer_adjusted_turn_indices'] = [4, 4]
        self.assert_invalid('duplicate customer', d)

    def test_already_defective_must_not_modify(self):
        d = copy.deepcopy(self.data)
        d['injection_metadata'].update({
            'already_defective': True, 'scope': 'pre_existing',
            'customer_adjusted_turn_indices': [],
        })
        self.assert_invalid('already_defective', d)

    def test_preexisting_no_edits(self):
        d = copy.deepcopy(self.data)
        d['injection_metadata'].update({
            'already_defective': True, 'scope': 'pre_existing',
            'customer_adjusted_turn_indices': [],
        })
        d['injected_conversation'] = copy.deepcopy(self.source)
        self.check(d)

    def test_turn_ids_can_be_nonconsecutive_if_preserved(self):
        d = copy.deepcopy(self.data)
        source = copy.deepcopy(self.source)
        for turn, tid in zip(source, [10, 20, 30, 40]):
            turn['turn_id'] = tid
        for turn, tid in zip(d['injected_conversation'], [10, 20, 30, 40]):
            turn['turn_id'] = tid
        d['injection_metadata']['injected_turn_indices'] = [30]
        d['injection_metadata']['customer_adjusted_turn_indices'] = [40]
        d['judge_evaluations']['task_closure']['violations'][0]['turn_id'] = 30
        self.check(d, source)

    def test_turn_id_renumbering_rejected(self):
        d = copy.deepcopy(self.data)
        d['injected_conversation'][1]['turn_id'] = 5
        self.assert_invalid('turn_id', d)

    def test_scenario_cannot_change(self):
        d = copy.deepcopy(self.data)
        d['injection_metadata']['scenario_modified'] = True
        d['injection_metadata']['modified_scenario_fields'] = {'a': 1}
        self.assert_invalid('cannot modify scenario', d)

    def test_core_process_record_writes_customer_edit_metadata(self):
        input_record = {
            'conversation_id': 'demo-v54',
            'target_judge': 'task_closure',
            'sampled_params': self.params,
            'assembled_prompt': 'Generate one task closure failure.',
            'source_turns': self.source,
        }
        async def run_test():
            with tempfile.TemporaryDirectory() as td:
                output = Path(td) / 'golden.jsonl'
                failed = Path(td) / 'failed.jsonl'
                with patch('generate_golden_dataset.request_with_backoff',
                           new=AsyncMock(return_value=json.dumps(self.data))):
                    await process_record(
                        input_record, object(), 'chatgpt_oauth', 'gpt-6-luna',
                        asyncio.Semaphore(1), output, failed,
                        asyncio.Lock(), asyncio.Lock(), 15.0
                    )
                self.assertTrue(output.exists())
                self.assertFalse(failed.exists())
                persisted = json.loads(output.read_text(encoding='utf-8').strip())
                self.assertEqual(persisted['injection_metadata']['customer_adjusted_turn_indices'], [4])
        asyncio.run(run_test())

    def test_general_judge_rejects_task_closure_specific_metadata(self):
        d = copy.deepcopy(self.data)
        d['injection_metadata']['target_judge'] = 'intent_understanding'
        d['judge_evaluations']['intent_understanding'] = d['judge_evaluations'].pop('task_closure')
        with self.assertRaises(Exception) as ctx:
            validate_record(d, self.params, 'intent_understanding')
        self.assertIn('customer_adjusted_turn_indices', str(ctx.exception))


if __name__ == '__main__':
    unittest.main()
