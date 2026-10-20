import json
from pathlib import Path
import unittest
from unittest.mock import patch
import alias_supervision_preflight_20261020 as t

ROOT=Path(__file__).resolve().parents[1]


class Inputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files,cls.m,cls.review=t.compose(ROOT)

    def test_old_real_preflight_is_verified_and_not_training(self):
        report=t.audit_previous(ROOT)
        self.assertEqual(report['inference_checks'],1324)
        self.assertEqual(sum(v['unknown_zero_gradient'] for v in report['loss_checks']),37)
        self.assertFalse(report['training_ready'])

    def test_all_questions_and_targets_are_identical_to_v226(self):
        for name,values in self.files.items():
            old=t.rows(ROOT/t.old.FOLDER/(name+'.jsonl'))
            self.assertEqual([r['messages'] for r in values],[r['messages'] for r in old])
        self.assertEqual(self.m['paired_changed_training_targets'],6)
        self.assertEqual(self.m['new_positive_relations'],0)
        self.assertEqual(self.m['new_supervision_sources'],0)

    def test_alias_masks_match_between_arms_and_do_not_cover_positive_bits(self):
        for a,b in zip(self.files['control_train'],self.files['repair_train']):
            ca,cb=[json.loads(r['channel']) for r in (a,b)]
            for key in ('unknown_entity_ids','unknown_relation_ids'):
                self.assertEqual(ca[key],cb[key])
            for r in (a,b):
                c=json.loads(r['channel']);s=t.policy.selection_for(r['messages'][1]['content'],r['messages'][2]['content'])
                self.assertFalse(set(c['unknown_entity_ids']) & set(s['selected_entity_ids']))
                self.assertFalse(set(c['unknown_relation_ids']) & set(s['selected_relation_ids']))
        self.assertGreater(self.m['unknown_entity_counts']['train'],0)
        self.assertGreater(self.m['unscored_counts']['train'],16)

    def test_alias_review_is_explicit_source_not_identifier_rule(self):
        self.assertTrue(self.review['cases'])
        self.assertTrue(all(c['review']['groups'] and c['source_sha256'] for c in self.review['cases']))
        self.assertEqual(self.review['policy']['status'],'user_confirmed_conservative_alias_quarantine')


class ReportGuards(unittest.TestCase):
    def test_failed_real_report_cannot_pass_audit(self):
        original=t.read
        def damaged(path):
            value=original(path)
            if path.name=='previous_preflight_report.json':value['passed']=False
            return value
        with patch.object(t,'read',side_effect=damaged):
            with self.assertRaises(AssertionError):t.audit_previous(ROOT)


if __name__=='__main__':unittest.main()
