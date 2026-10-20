import copy
import json
import unittest
from ecospec_kg import alias_partial_label_v227 as p


class Tokenizer:
    def __call__(self, text, **kwargs):
        return {'input_ids': [ord(c) for c in text], 'offset_mapping': [(i, i+1) for i in range(len(text))]}


class AliasReview(unittest.TestCase):
    def setUp(self):
        self.unit = {'provenance': {'evidence_spans': [{'span_id': 'w', 'text': '坡长可从地面模型（GM）提取。'}]}}
        self.graph = {'entities': [dict(entity_id=i, name=n, entity_type=t) for i,n,t in
                       [('full','地面模型（GM）','data_source'), ('cn','地面模型','data_source'),
                        ('latin','GM','data_source'), ('wrongtype','GM','method'), ('head','坡长','observation_variable')]],
                      'relations': [dict(relation_id=i,head_id='head',tail_id=t,relation_type='sourced_from')
                                    for i,t in [('r1','full'),('r2','cn'),('r3','latin')]]}
        self.selection = {'selected_entity_ids':['head','full'], 'selected_relation_ids':['r1']}

    def test_explicit_aliases_unknown_and_positive_graph_preserved(self):
        before=copy.deepcopy(self.graph)
        review=p.review_aliases(self.unit,self.graph,[self.selection])
        self.assertEqual(review['unknown_entity_ids'],['cn','latin'])
        self.assertEqual(review['unknown_relation_ids'],['r2','r3'])
        self.assertEqual(self.graph,before)
        self.assertEqual(review['groups'][0]['source_span_ids'],['w'])

    def test_no_source_witness_or_positive_anchor_means_no_relabel(self):
        unit=copy.deepcopy(self.unit);unit['provenance']['evidence_spans'][0]['text']='模型可以提取坡长。'
        self.assertFalse(p.review_aliases(unit,self.graph,[self.selection])['unknown_entity_ids'])
        empty={'selected_entity_ids':[], 'selected_relation_ids':[]}
        self.assertFalse(p.review_aliases(self.unit,self.graph,[empty])['unknown_entity_ids'])

    def test_pair_masks_use_union_of_existing_positive_anchors(self):
        other={'selected_entity_ids':['head','cn'], 'selected_relation_ids':['r2']}
        review=p.review_aliases(self.unit,self.graph,[self.selection,other])
        self.assertEqual(review['unknown_entity_ids'],['latin'])
        self.assertEqual(review['unknown_relation_ids'],['r3'])

    def test_verified_negative_not_silently_quarantined(self):
        review=p.review_aliases(self.unit,self.graph,[self.selection],confirmed_negative_ids=['r2'])
        self.assertEqual(review['unknown_relation_ids'],['r3'])
        self.assertEqual(review['retained_confirmed_negative_ids'],['r2'])

    def test_two_member_ascii_spaced_alias_is_supported(self):
        self.graph['entities']=[e for e in self.graph['entities'] if e['entity_id']!='cn']
        self.graph['relations']=[r for r in self.graph['relations'] if r['relation_id']!='r2']
        self.unit['provenance']['evidence_spans'][0]['text']='坡长可从地面模型 ( G M ) 提取。'
        review=p.review_aliases(self.unit,self.graph,[self.selection])
        self.assertEqual(review['unknown_entity_ids'],['latin'])


class CombinedMask(unittest.TestCase):
    def setUp(self):
        self.prompt=json.dumps({'candidate_entities':[{'id':'a'},{'id':'b'}],
                                'candidate_relations':[{'id':'r1'},{'id':'r2'}]})
        self.answer=p.answer_for(self.prompt,{'selected_entity_ids':['a'],'selected_relation_ids':['r1']})

    def test_entity_and_relation_unknown_bits_have_no_label(self):
        channel=p.supervision_channel(self.prompt,self.answer,['r2'],['b'])
        ids=[9]+[ord(c) for c in self.answer]+[3]
        encoded={'input_ids':ids,'labels':[-100]+ids[1:]}
        result,proof=p.mask_encoded(encoded,Tokenizer(),self.prompt,self.answer,channel)
        self.assertEqual(proof['unknown_count'],2)
        self.assertEqual(len(proof['unknown_entity_token_indices']),1)
        self.assertEqual(len(proof['unknown_relation_token_indices']),1)
        self.assertTrue(all(result['labels'][i]==-100 for i in proof['unknown_token_indices']))
        self.assertTrue(all(result['labels'][i]==ids[i] for i in proof['known_token_indices']))
        self.assertEqual(encoded['labels'][1:],ids[1:])

    def test_bad_unknown_entity_or_old_metadata_is_rejected(self):
        with self.assertRaises(ValueError):p.supervision_channel(self.prompt,self.answer,[],['absent'])
        ids=[ord(c) for c in self.answer]
        with self.assertRaises(ValueError):p.mask_encoded({'input_ids':ids,'labels':ids},Tokenizer(),self.prompt,self.answer,'{}')

    def test_flipping_unknown_entity_and_relation_preserves_all_supervised_labels(self):
        def encoded(answer):
            ids=[9]+[ord(c) for c in answer]+[3]
            return {'input_ids':ids,'labels':[-100]+ids[1:]}
        other=json.loads(self.answer);other['entity_decisions']='1 1';other['relation_decisions']='1 1'
        other=json.dumps(other,separators=(',',':'))
        original,proof=p.mask_encoded(encoded(self.answer),Tokenizer(),self.prompt,self.answer,
                                      p.supervision_channel(self.prompt,self.answer,['r2'],['b']))
        changed,again=p.mask_encoded(encoded(other),Tokenizer(),self.prompt,other,
                                     p.supervision_channel(self.prompt,other,['r2'],['b']))
        self.assertEqual(original['labels'],changed['labels'])
        self.assertEqual(proof,again)


if __name__=='__main__':unittest.main()
