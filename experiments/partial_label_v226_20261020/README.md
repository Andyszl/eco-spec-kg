# Partial-label CPU preflight, 2026-10-20

The received server inspection was successful: result commit
`407cdf3ef07d6d500a2c36d1a563028ded73a35d`, ms-swift 4.4.2,
40 source files, 705769 bytes. All received file hashes were checked.
`runtime_source_lock.json` pins the eight interfaces used here. This does not
claim the separate `swift/loss` and `swift/loss_scale` directories were collected.
The plugin uses the observed Template labels API and binary default loss route,
not an unverified custom LossScale API. No installed SDK files are patched.

## Changes and experimental scope

- Rebuild 626 control / 626 repair / 72 shared internal-validation records from
  the locked complete graphs and reviewed splits. No dev/test data builds inputs.
- The question includes every candidate and never includes the unknown-ID list.
  The same question and decision schema are used in inference.
- Two spaced binary strings specify entity and relation decisions in the original
  candidate order. There is no old selected-ID output schema in the question.
  A strict parser converts decisions back to source-local IDs and original graph
  members; no relation names, graph members or evidence are invented.
- Old selected_entity_ids meant standalone entities, with relation endpoints
  restored automatically. Dense entity targets encode the original **effective
  graph membership**, including those endpoints. Otherwise supported endpoints
  would receive false zero labels. This is a representation conversion, not new
  semantic annotation. Selected graphs are preserved except the shared name fix.
- Unknown relation bits are zero placeholders in teacher forcing. The SDK-supported
  `channel` column carries the supervision metadata, outside the question. A new
  template checks its prompt/answer hashes, sets only those bit-token labels to
  -100, and removes the channel before model collation. Known positive and
  retained unselected decisions keep labels; no unknown is newly adjudicated.
- Token alignment is exact, with one token per individual decision (optional
  surrounding whitespace only). Merged decisions, missing metadata, truncation,
  missing labels and a mismatch in answers all stop the check.
- Apply the separately saved assistant-reviewed method boundary correction to
  both training arms. Original frozen inputs and historical results stay intact.
  The six pre-existing paired supervision differences are preserved.

## Verification and next server action

Local tests use a character tokenizer and a minimal SDK double. They are not a
real Qwen tokenizer, actual SDK execution, model training or a new dev experiment.
The CPU wrapper must run on the user's existing server environment. It checks
the installed SDK sources, tokenizer/config hashes, strict dataset loading and
record order, every mask after template encoding and collation, and actual
inference/training prompt-prefix equality. It also flips unknown placeholders
and checks the retained labels do not change.

For each record containing unknown bits, the check calls the real SDK
Seq2SeqTrainer.compute_loss and Template.compute_sft_loss with synthetic
three-class logits. It verifies the causal shift gives zero direct gradient to
unknown bits and nonzero gradient to known bits. This is a synthetic pipeline
test, **not a Qwen forward or a GPU smoke-training test**.

Run `tools/run_partial_label_preflight_20261020.sh` in the fixed new checkout.
It hides GPUs, uses local tokenizer files, never starts training or deployment,
creates a new results directory, and uploads reports/logs through a separate
Git result branch (three transfer retries, no force or TLS verification bypass).
Failure reports preserve failure status and must be reviewed before any rerun.

## Remaining limits; training is blocked

- A masked placeholder still conditions later teacher-forced decisions. Direct
  negative loss is removed; this is not marginalization over unknown decisions.
- Changing to dense decisions changes the objective and output format. Older dev
  metrics cannot be reused to claim improvement or isolate masking's contribution.
  Future base/control/repair comparisons must use the same new protocol.
- No new source positives or independent validation cases were added. Coverage
  counts distinguish inherited unselected targets from verified negatives. Shared
  source hashes are explicitly reported; internal validation is not independent.
- There is no training launcher in this delivery. `TRAINING_READY=false` remains
  true even if the technical CPU check passes. Review the actual report and source
  coverage before preparing any fixed seed42 pair. The original strict dev gate,
  three-seed restriction and final independent-source evaluation remain in force.
