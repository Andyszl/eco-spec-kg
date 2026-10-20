"""Add one template through Swift's registration API; no installed files patched."""
from copy import deepcopy
import importlib.metadata
import json

from swift.template import register_template
from swift.template.register import TEMPLATE_MAPPING
from swift.template.templates.qwen import Qwen3_5Template
from ecospec_kg import partial_label_v226 as policy

assert importlib.metadata.version('ms-swift') == '4.4.2', 'Unsupported Swift runtime'


class PartialLabelTemplate(Qwen3_5Template):
    def _encode(self, inputs):
        # Capture before Qwen adds its non-thinking prefix to messages.
        prompt = inputs.messages[0]['content'] if inputs.messages else None
        answer = inputs.messages[-1]['content'] if inputs.messages and inputs.messages[-1]['role'] == 'assistant' else None
        channel = inputs.channel
        encoded = super()._encode(inputs)
        if self.is_training:
            if not channel or answer is None or len(inputs.messages) != 2:
                raise ValueError('Partial-label training requires exactly one question, answer and mask metadata')
            encoded, _ = policy.mask_encoded(encoded, self.tokenizer, prompt, answer, channel)
        elif channel is not None:
            raise ValueError('Inference cannot carry supervision metadata')
        return encoded

    def encode(self, *args, **kwargs):
        encoded = super().encode(*args, **kwargs)
        if not isinstance(encoded, dict):
            raise ValueError('Partial-label protocol forbids splitting or packing a question')
        # Channel is an SDK-supported data column, used only to transport the
        # mask to _encode. It is not a metric channel or an input to the model.
        encoded.pop('channel', None)
        return encoded


meta = deepcopy(TEMPLATE_MAPPING['qwen3_5'])
meta.template_type = policy.TEMPLATE
meta.template_cls = PartialLabelTemplate
register_template(meta)
