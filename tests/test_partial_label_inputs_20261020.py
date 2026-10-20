import json
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch

import partial_label_preflight_20261020 as tool
from ecospec_kg import partial_label_v226 as p

ROOT = Path(__file__).resolve().parents[1]


class Inputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files, cls.manifest = tool.compose(ROOT)

    def test_split_masks_and_pair_are_preserved(self):
        self.assertEqual({n: len(v) for n, v in self.files.items()},
                         {'control_train': 626, 'repair_train': 626, 'validation': 72})
        m = self.manifest
        self.assertEqual(m['unscored_counts'], {'train': 16, 'validation': 5})
        self.assertEqual(m['paired_changed_training_targets'], 6)
        self.assertEqual(m['name_patch_records'], 2)
        self.assertFalse(m['training_ready'])
        self.assertEqual(m['new_supervision_sources'], 0)
        for a, b in zip(self.files['control_train'], self.files['repair_train']):
            self.assertEqual(a['messages'][:2], b['messages'][:2])
            self.assertEqual(json.loads(a['channel'])['unknown_relation_ids'], json.loads(b['channel'])['unknown_relation_ids'])
        for values in self.files.values():
            for row in values:
                q = json.loads(row['messages'][1]['content'])
                self.assertNotIn('decision_scope', q)
                self.assertNotIn('output_schema', q)
                self.assertNotIn('unknown_relation_ids', q)
                self.assertEqual(len(p.selection_for(row['messages'][1]['content'], row['messages'][2]['content'])), 2)

    def test_name_patch_affects_both_arms_only(self):
        for n in ('control_train', 'repair_train'):
            row = next(r for r in self.files[n] if tool.uid(r) == '1acaf3c9cad2f54d')
            selection = p.selection_for(row['messages'][1]['content'], row['messages'][2]['content'])
            self.assertIn('42c3823dbe0cd908', selection['selected_entity_ids'])
            self.assertNotIn('81677f8ab30adb93', selection['selected_entity_ids'])
            self.assertEqual(selection['selected_relation_ids'], ['ac9ea1b41ca52471'])

    def test_received_sdk_sources_are_pinned(self):
        lock = tool.verify_runtime_lock(ROOT)
        self.assertEqual(lock['code_commit'], '328bd7d2c14396e50bd65e29cfb26891b69f86cd')
        self.assertIn('swift/template/base.py', lock['files'])
        self.assertIn('swift/trainers/seq2seq_trainer.py', lock['files'])


class RuntimeGuards(unittest.TestCase):
    def test_loader_null_schema_fields_allowed_but_loss_changes_rejected(self):
        original = {'messages': [{'role': 'user', 'content': 'same'}], 'channel': 'mask'}
        loaded = {'messages': [{'role': 'user', 'content': 'same', 'loss': None, 'loss_scale': None}], 'channel': 'mask'}
        tool.check_loaded_row(original, loaded)
        loaded['messages'][0]['loss'] = False
        with self.assertRaises(AssertionError):
            tool.check_loaded_row(original, loaded)

    def test_visible_gpu_blocks_before_creating_report(self):
        with tempfile.TemporaryDirectory() as d, patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '0'}):
            out = Path(d) / 'job'
            with self.assertRaises(AssertionError):
                tool.preflight(ROOT, Path(d) / 'no-model', out)
            self.assertFalse(out.exists())

    def test_existing_output_preserved_before_sdk_import(self):
        with tempfile.TemporaryDirectory() as d, patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': ''}):
            out = Path(d); marker = out / 'original.txt'; marker.write_bytes(b'preserve')
            with self.assertRaises(AssertionError):
                tool.preflight(ROOT, Path(d) / 'no-model', out)
            self.assertEqual(marker.read_bytes(), b'preserve')


if __name__ == '__main__':
    unittest.main()
