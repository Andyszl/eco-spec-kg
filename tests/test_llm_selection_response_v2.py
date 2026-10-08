from __future__ import annotations

import json
import unittest

from ecospec_kg.extractor_v2 import _parse_llm_selection


class LLMSelectionResponseTests(unittest.TestCase):
    def test_repeated_selection_ids_keep_first_occurrence_order(self) -> None:
        response = json.dumps(
            {
                "selected_entity_ids": ["entity-b", 17, "entity-a", "17", "entity-b"],
                "selected_relation_ids": [
                    "ba2680accdff433c",
                    "d18589b2aec60368",
                    "ba2680accdff433c",
                    "da0687cf2a5c71c5",
                ],
            }
        )
        self.assertEqual(
            _parse_llm_selection(response),
            {
                "selected_entity_ids": ["entity-b", "17", "entity-a"],
                "selected_relation_ids": [
                    "ba2680accdff433c",
                    "d18589b2aec60368",
                    "da0687cf2a5c71c5",
                ],
            },
        )

    def test_server_response_with_empty_think_prefix(self) -> None:
        response = (
            '<think>\n\n</think>\n\n{"selected_entity_ids": [], '
            '"selected_relation_ids": ["6b1b371a25cf8297"]}'
        )
        self.assertEqual(
            _parse_llm_selection(response),
            {
                "selected_entity_ids": [],
                "selected_relation_ids": ["6b1b371a25cf8297"],
            },
        )

    def test_reasoning_json_is_not_used_as_final_selection(self) -> None:
        response = (
            '<think>Consider this draft:\n```json\n'
            '{"selected_entity_ids": ["draft"], "selected_relation_ids": []}'
            '\n```\nUse the final answer below.</think>\n'
            '```json\n{"selected_entity_ids": ["final"], '
            '"selected_relation_ids": []}\n```'
        )
        self.assertEqual(
            _parse_llm_selection(response)["selected_entity_ids"], ["final"]
        )

    def test_existing_json_and_fenced_json_remain_supported(self) -> None:
        payload = {"selected_entity_ids": ["entity"], "selected_relation_ids": []}
        content = json.dumps(payload)
        for response in (
            content,
            f"```json\n{content}\n```",
            f"```\n{content}\n```",
            f"  <think></think>\n{content}\n ",
        ):
            with self.subTest(response=response):
                self.assertEqual(_parse_llm_selection(response), payload)

    def test_think_markers_inside_json_are_preserved(self) -> None:
        payload = {
            "selected_entity_ids": ["<think>entity</think>"],
            "selected_relation_ids": [],
        }
        self.assertEqual(_parse_llm_selection(json.dumps(payload)), payload)

    def test_incomplete_or_invalid_answer_remains_an_error(self) -> None:
        responses = (
            '<think>\n```json\n{"selected_entity_ids": ["draft"]}\n```',
            '<think>unfinished {"selected_entity_ids": []}',
            '<think>reasoning</think>',
            '<think></think>\n{"selected_entity_ids": [',
            'Explanation: {"selected_entity_ids": []}',
        )
        for response in responses:
            with self.subTest(response=response):
                with self.assertRaises(ValueError):
                    _parse_llm_selection(response)


if __name__ == "__main__":
    unittest.main()
