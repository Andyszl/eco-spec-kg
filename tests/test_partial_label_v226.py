import copy
import json
import unittest

from ecospec_kg import partial_label_v226 as p


class CharTokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(c) for c in text],
                "offset_mapping": [(i, i + 1) for i in range(len(text))]}


class PartialLabels(unittest.TestCase):
    def setUp(self):
        self.prompt = json.dumps({"candidate_entities": [{"id": "e1"}, {"id": "e2"}],
                                  "candidate_relations": [{"id": "r1"}, {"id": "r2"}, {"id": "r3"}]})
        self.selection = {"selected_entity_ids": ["e1"], "selected_relation_ids": ["r1"]}
        self.answer = p.answer_for(self.prompt, self.selection)
        self.channel = p.supervision_channel(self.prompt, self.answer, ["r2"])

    def encoded(self, answer=None):
        answer = answer or self.answer
        ids = [9, 10] + [ord(c) for c in answer] + [3]
        return {"input_ids": ids, "labels": [-100, -100] + ids[2:], "loss_scale": [0, 0] + [1] * (len(ids) - 2)}

    def test_round_trip_and_strict_schema(self):
        self.assertEqual(p.selection_for(self.prompt, self.answer), self.selection)
        for bad in ['{}', self.answer + ' trailing', self.answer.replace('1 0', '1 2'),
                    self.answer.replace('1 0 0', '1 0')]:
            with self.assertRaises(ValueError):
                p.selection_for(self.prompt, bad)

    def test_relation_endpoints_are_positive_members_not_false_entity_targets(self):
        graph = {'entities': [{'entity_id': 'e1'}, {'entity_id': 'e2'}],
                 'relations': [{'relation_id': 'r1', 'head_id': 'e1', 'tail_id': 'e2'}]}
        old = {'selected_entity_ids': [], 'selected_relation_ids': ['r1']}
        self.assertEqual(p.effective_selection(graph, old),
                         {'selected_entity_ids': ['e1', 'e2'], 'selected_relation_ids': ['r1']})
        self.assertEqual(old['selected_entity_ids'], [])

    def test_only_unknown_decision_token_is_masked(self):
        original = self.encoded(); before = copy.deepcopy(original)
        masked, proof = p.mask_encoded(original, CharTokenizer(), self.prompt, self.answer, self.channel)
        self.assertEqual(original, before)
        self.assertEqual(len(proof['unknown_token_indices']), 1)
        unknown = proof['unknown_token_indices'][0]
        self.assertEqual(masked['labels'][unknown], -100)
        self.assertEqual(masked['loss_scale'][unknown], 0)
        self.assertTrue(all(masked['labels'][i] == masked['input_ids'][i] for i in proof['known_token_indices']))
        self.assertEqual([i for i, (a, b) in enumerate(zip(before['labels'], masked['labels'])) if a != b], [unknown])

    def test_counterfactual_unknown_is_not_scored(self):
        other = json.loads(self.answer); other['relation_decisions'] = '1 1 0'
        other = json.dumps(other, separators=(',', ':'))
        channel = p.supervision_channel(self.prompt, other, ['r2'])
        _, first = p.mask_encoded(self.encoded(), CharTokenizer(), self.prompt, self.answer, self.channel)
        _, second = p.mask_encoded(self.encoded(other), CharTokenizer(), self.prompt, other, channel)
        self.assertEqual(first['unknown_token_indices'], second['unknown_token_indices'])
        self.assertEqual(first['known_token_indices'], second['known_token_indices'])

    def test_unknown_positive_and_invalid_metadata_rejected(self):
        with self.assertRaises(ValueError):
            p.supervision_channel(self.prompt, self.answer, ['missing'])
        for channel in ['{}', self.channel.replace('r2', 'r9')]:
            with self.assertRaises(ValueError):
                p.mask_encoded(self.encoded(), CharTokenizer(), self.prompt, self.answer, channel)
        with self.assertRaises(ValueError):
            p.mask_encoded(self.encoded(), CharTokenizer(), self.prompt + ' ', self.answer, self.channel)

    def test_missing_duplicate_or_already_masked_answer_is_rejected(self):
        for encoded in [dict(input_ids=[1], labels=[-100]),
                        dict(input_ids=[ord(c) for c in self.answer * 2], labels=[ord(c) for c in self.answer * 2]),
                        dict(input_ids=self.encoded()['input_ids'], labels=[-100] * len(self.encoded()['labels']))]:
            with self.assertRaises(ValueError):
                p.mask_encoded(encoded, CharTokenizer(), self.prompt, self.answer, self.channel)

    def test_token_merging_across_decisions_is_rejected(self):
        class Merged(CharTokenizer):
            def __call__(self, text, **kwargs):
                out = super().__call__(text, **kwargs)
                start = text.index('1 0 0')
                out['offset_mapping'][start] = (start, start + 3)
                return out
        with self.assertRaises(ValueError):
            p.mask_encoded(self.encoded(), Merged(), self.prompt, self.answer, self.channel)


if __name__ == '__main__':
    unittest.main()
