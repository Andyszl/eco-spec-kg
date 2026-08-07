from __future__ import annotations

import json
import io
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

from ecospec_kg.io_utils import read_json, write_jsonl
from ecospec_kg.providers import CompletionHTTPError, OpenAICompatibleProvider
from ecospec_kg.training import LoRASettings, build_swift_command, run_swift_lora


class QwenRuntimeTests(unittest.TestCase):
    def test_openai_provider_sends_qwen_non_thinking_setting(self) -> None:
        response = MagicMock()
        response.read.return_value = json.dumps(
            {"choices": [{"message": {"content": '{"relations": []}'}}]}
        ).encode("utf-8")
        response.__enter__.return_value = response

        provider = OpenAICompatibleProvider(
            base_url="http://127.0.0.1:8000/v1",
            api_key="local-token",
            model="Qwen3.5-9B",
            enable_thinking=False,
        )
        with patch("urllib.request.urlopen", return_value=response) as urlopen:
            self.assertEqual(provider.complete("system", "prompt"), '{"relations": []}')

        request = urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["model"], "Qwen3.5-9B")
        self.assertEqual(
            payload["chat_template_kwargs"], {"enable_thinking": False}
        )

    def test_openai_provider_preserves_http_error_body(self) -> None:
        provider = OpenAICompatibleProvider(
            base_url="http://127.0.0.1:8000/v1",
            api_key="local-token",
            model="Qwen3.5-9B",
        )
        error = HTTPError(
            url="http://127.0.0.1:8000/v1/chat/completions",
            code=400,
            msg="Bad Request",
            hdrs=None,
            fp=io.BytesIO(b'{"error":"context length exceeded"}'),
        )
        with patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaisesRegex(CompletionHTTPError, "context length exceeded"):
                provider.complete("system", "prompt")
        self.assertEqual(provider.last_raw_response["http_status"], 400)
        self.assertIn(
            "context length exceeded", provider.last_raw_response["error_body"]
        )

    def test_swift_command_uses_text_only_qwen35_settings(self) -> None:
        settings = LoRASettings(
            model_name="/work/models/Qwen3.5-9B",
            trainer="swift",
            precision="bf16",
        )
        command = build_swift_command(
            Path("train.jsonl"), Path("runs/smoke"), settings
        )
        rendered = " ".join(command)
        self.assertIn("--torch_dtype bfloat16", rendered)
        self.assertIn("--freeze_vit true", rendered)
        self.assertIn("--freeze_aligner true", rendered)
        self.assertIn("--add_non_thinking_prefix true", rendered)
        self.assertIn("--target_modules all-linear", rendered)
        self.assertIn("--padding_free false", rendered)
        self.assertIn("--packing false", rendered)
        self.assertIn("--sequence_parallel_size 1", rendered)

    def test_swift_smoke_run_writes_limited_data_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training_path = root / "training.jsonl"
            output_dir = root / "run"
            write_jsonl(
                training_path,
                [
                    {"messages": [{"role": "assistant", "content": str(i)}]}
                    for i in range(3)
                ],
            )
            calls: list[list[str]] = []

            def runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
                calls.append(command)
                return subprocess.CompletedProcess(command, 0)

            manifest = run_swift_lora(
                training_path,
                output_dir,
                LoRASettings(trainer="swift"),
                smoke_limit=2,
                runner=runner,
            )

            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["training_records"], 2)
            self.assertTrue((output_dir / "smoke_training.jsonl").exists())
            self.assertEqual(read_json(output_dir / "run_manifest.json")["status"], "complete")
            self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
