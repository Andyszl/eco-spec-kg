import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import final_source_preflight_20261021 as m

ROOT = Path(__file__).resolve().parents[1]


class FinalSourcePreflightTests(unittest.TestCase):
    def test_reviewed_graph_has_mechanical_inverse_and_valid_endpoints(self):
        packet = m.verify_packet(ROOT)
        self.assertEqual(len(packet['units']), 2)
        self.assertEqual([len(x['entities']) for x in packet['gold']], [11, 10])
        self.assertEqual([len(x['relations']) for x in packet['gold']], [7, 10])
        g = packet['gold'][0]
        self.assertTrue(any(r['head_name'] == 'WEQI' and r['relation_type'] == 'calculated_by'
                            and r['tail_name'] == '公式（1）' for r in g['relations']))
        for row in packet['gold']:
            ids = {e['entity_id'] for e in row['entities']}
            self.assertTrue(all(r['head_id'] in ids and r['tail_id'] in ids for r in row['relations']))
        self.assertEqual(packet['protocol']['reviewer']['name'], '沈老师')

    def test_unit_inputs_are_source_transcription_not_annotation(self):
        units = m.verify_packet(ROOT)['units']
        raw = json.dumps(units, ensure_ascii=False)
        for forbidden in ('gold_annotation', 'selected_entity_ids', 'manual_variable_roles', 'review_status', 'unknown_relation_ids'):
            self.assertNotIn(forbidden, raw)
        self.assertIn('sum(Z(X_i)*λ_i,i,1,n)', raw)
        self.assertIn('1/d_i^4', raw)
        self.assertNotIn('4.2.3', raw)

    def test_source_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copytree(ROOT / m.FOLDER, root / m.FOLDER)
            (root / m.FOLDER / 'sources/HJ1339-2023.pdf').write_bytes(b'%PDF-tampered')
            with self.assertRaisesRegex(AssertionError, 'Changed material'):
                m.verify_packet(root)

    def test_unknown_scope_is_preserved_and_not_complete_gold(self):
        p = m.verify_packet(ROOT)['protocol']
        self.assertFalse(p['exhaustive_full_source_gold'])
        self.assertFalse(p['final_inference_ready'])
        self.assertEqual(p['data_source_extension_support'], 0)
        self.assertEqual(p['indicator_rooted_formula_cases'], 1)
        self.assertEqual(p['variable_rooted_formula_cases'], 2)
        self.assertTrue(p['unresolved_scope'])

    def test_gpu_visible_or_existing_output_prevents_sdk_loading(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '0'}):
            with self.assertRaisesRegex(AssertionError, 'CPU only'):
                m.preflight(ROOT, Path(tmp), Path(tmp) / 'out')
        with tempfile.TemporaryDirectory() as tmp, patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': ''}):
            with self.assertRaisesRegex(AssertionError, 'Preserve'):
                m.preflight(ROOT, Path(tmp), Path(tmp))

    def test_frozen_candidate_diagnostic_is_not_a_new_model_result(self):
        p = m.verify_packet(ROOT)
        d = m.candidate_diagnostic(p['units'], p['gold'])
        self.assertEqual(d['model_calls'], 0)
        self.assertEqual(len(d['records']), 2)
        self.assertTrue(all(r['gold_relations'] > 0 for r in d['records']))
        self.assertFalse(d['model_performance_measured'])


if __name__ == '__main__':
    unittest.main()
