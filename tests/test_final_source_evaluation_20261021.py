import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import final_source_evaluation_20261021 as m
from ecospec_kg import candidate_profile_v228 as profile
from ecospec_kg.experiment_io_v2 import sha256_json

ROOT = Path(__file__).resolve().parents[1]


class FinalEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.packet, self.lock = m.verify_inputs(ROOT)
        self.units, self.gold = self.packet['units'], self.packet['gold']

    def test_actual_server_preflight_bound_to_exact_two_prompts(self):
        self.assertEqual(self.lock['maximum_prompt_plus_output'], 5255)
        self.assertEqual([r['prompt_tokens'] for r in self.lock['records']], [2772, 5141])

    def test_positive_only_metrics_never_invent_negatives(self):
        result = m.partial_scores(self.gold, self.gold)
        self.assertEqual(result['entities']['recovered_known_positive'], 21)
        self.assertEqual(result['relations']['recovered_known_positive'], 17)
        self.assertEqual(result['relations']['known_positive_recall'], 1)
        for kind in ('entities', 'relations'):
            for field in ('false_positive', 'precision', 'f1'):
                self.assertIsNone(result[kind][field])

    def test_extra_predictions_remain_visible_unadjudicated(self):
        predictions = copy.deepcopy(self.gold)
        extra = copy.deepcopy(predictions[1]['relations'][0])
        extra['relation_type'] = 'sourced_from'
        predictions[1]['relations'].append(extra)
        result = m.partial_scores(self.gold, predictions)
        self.assertEqual(result['relations']['unadjudicated_extra_count'], 1)
        self.assertEqual(result['relations']['recovered_known_positive'], 17)
        self.assertIsNone(result['relations']['false_positive'])
        self.assertEqual(result['relations']['unadjudicated_extra_keys'][0][3], 'sourced_from')

    def test_indicator_and_variable_paths_separate_zero_support_null(self):
        result = m.paths(self.units, self.gold, self.gold)
        self.assertEqual(result['indicator']['scorable_paths'], 1)
        self.assertEqual(result['variable']['support'], 2)
        self.assertEqual(result['variable']['complete'], 2)
        self.assertEqual(result['variable_method_extension']['support'], 1)
        self.assertEqual(result['variable_method_extension']['complete'], 1)
        for key in ('indicator_method_extension', 'data_source_extension'):
            self.assertEqual(result[key]['support'], 0)
            self.assertIsNone(result[key]['complete'])
            self.assertIsNone(result[key]['rate'])

    def test_structure_cannot_hide_missing_indicator_candidate(self):
        before = [m.empty_prediction(u) for u in self.units]
        modes = m.postprocess(self.units, before)
        result = m.paths(self.units, self.gold, modes['structure']['predictions'])
        self.assertEqual(result['indicator']['complete_paths'], 0)
        self.assertEqual(sum(len(p['relations']) for p in modes['structure']['predictions']), 15)
        self.assertEqual(before, [m.empty_prediction(u) for u in self.units])

    def test_evidence_preserves_semantic_selection(self):
        before = [m.empty_prediction(u) for u in self.units]
        modes = m.postprocess(self.units, before)
        for left, right in [('raw', 'evidence'), ('structure', 'structure_evidence')]:
            for a, b in zip(modes[left]['predictions'], modes[right]['predictions']):
                self.assertEqual(m.semantic_keys(a), m.semantic_keys(b))

    def test_tampered_report_is_rejected_before_any_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / m.PREFLIGHT
            target.parent.mkdir(parents=True)
            target.write_text('{}', encoding='utf8')
            with self.assertRaisesRegex(AssertionError, 'Preflight'):
                m.verify_inputs(root)

    def test_six_requests_and_twelve_reports_no_extra_calls(self):
        calls = []
        class FakeProvider:
            def __init__(self, **kwargs): self.last_raw_response = None
            def complete(self, system, prompt):
                calls.append((self.model, sha256_json({'system':system,'prompt':prompt})))
                answer = m.scope.answer_for(prompt, {'selected_entity_ids':[],'selected_relation_ids':[]})
                self.last_raw_response = {'fixture':True, 'content':answer}
                return answer
        with tempfile.TemporaryDirectory() as tmp, patch.object(m.prior, 'Provider', FakeProvider):
            result = m.evaluate(ROOT, Path(tmp)/'evaluation', 'http://fixture', {})
            self.assertEqual(len(calls), 6)
            self.assertEqual(result['model_requests'], 6)
            self.assertEqual(sum(len(v) for v in result['results'].values()), 12)
            self.assertFalse(result['three_seed_eligible'])
            self.assertFalse(result['complete_precision_f1_available'])
            self.assertEqual(len({c[1] for c in calls}),2)
            for prompt in {c[1] for c in calls}:
                self.assertEqual(len({c[0] for c in calls if c[1] == prompt}),3)

    def test_missing_endpoint_or_role_breaks_path(self):
        prediction = copy.deepcopy(self.gold)
        prediction[1]['entities'] = [e for e in prediction[1]['entities'] if e['name'] != 'n']
        self.assertEqual(m.paths(self.units,self.gold,prediction)['variable']['complete'],0)
        prediction = copy.deepcopy(self.gold)
        prediction[1]['relations'] = [r for r in prediction[1]['relations'] if r['head_name'] != 'Z(B)' or r['relation_type'] != 'obtained_by']
        result = m.paths(self.units,self.gold,prediction)
        self.assertEqual(result['variable']['complete'],2)
        self.assertEqual(result['variable_method_extension']['complete'],0)

    def test_malformed_response_aborts_without_success_summary(self):
        class BadProvider:
            def __init__(self, **kwargs): self.last_raw_response = {'fixture':True}
            def complete(self, *args): return '{}'
        with tempfile.TemporaryDirectory() as tmp, patch.object(m.prior,'Provider',BadProvider):
            out = Path(tmp)/'evaluation'
            with self.assertRaises(ValueError):
                m.evaluate(ROOT,out,'http://fixture',{})
            self.assertFalse((out/'summary.json').exists())
            self.assertTrue((out/'base/extraction_failure.json').is_file())

    def test_gpu_guard_fails_closed_and_does_not_stop_existing_tasks(self):
        import subprocess
        for status, stdout in [(0, '12345\n'), (1, '')]:
            completed = subprocess.CompletedProcess([], status, stdout, 'query failed' if status else '')
            with patch.object(m.subprocess, 'run', return_value=completed):
                with self.assertRaisesRegex(AssertionError, 'GPU0'):
                    m.require_idle_gpu()


if __name__ == '__main__':
    unittest.main()
