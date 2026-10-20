# Conservative alias supervision revision, 2026-10-20

The v226 real CPU preflight has passed on the server. The original report is
preserved byte-for-byte as previous_preflight_report.json (SHA256
2a9b588ceb2a2e7ae3d6e5ebd03e783d6b138d766aaa022c762044a4434db891).
It verified 1324 records, max train lengths 7025/7025 and validation 5431,
1324 identical inference prefixes, and 37 unknown relation bits with zero direct
gradient in the real SDK loss route using synthetic logits. No Qwen weights or
model evaluation were used. It is not approval to train or evidence of new F1.

## User-confirmed policy and preserved data

The user confirmed: keep existing positive anchor names; unselected literal
aliases are unscored pending adjudication, rather than negative targets.
The assistant found four training units and one internal-validation unit with
literal Chinese-name/Latin-abbreviation aliases and mixed current targets.

The source-only review requires a same-type method/data-source candidate group,
an explicit parenthesized Latin alias in an original evidence span, and an
existing selected member. There are no source, entity or formula ID rules.
All existing positives from either paired arm are preserved. Both arms receive
the same unknown masks. Explicit reviewed negative relations are not silently
changed. No unselected alias becomes positive automatically.

Every question and assistant answer is byte-identical to v226. Complete graphs,
626/626/72 record membership/order, source provenance, the shared method-name
correction, the six paired target differences, old inputs and historical results
are retained. Only supervision metadata and token masks change. The question and
output format remain v226; the new template/policy version records mask changes.

Entity aliases must be masked too: masking only their relation decisions would
still train the alias entities as negatives. The new policy extends the existing
strict, exact one-token-per-decision alignment to unknown entity bits, while
keeping known decision labels. Labels/metadata are checked through the real SDK
loader, template, collator and synthetic loss route in the new CPU preflight.
Neither installed SDK files nor model weights are modified.

## Expected counts; server confirmation pending

Training: 8 alias entity decisions and 8 alias relation decisions per arm are
newly unscored. Total unscored relations become 24 per arm, including the old 16.
Internal validation: 2 alias entity decisions and 2 alias relation decisions
are newly unscored; total unscored relations become 7, including the old 5.
Across both arms and validation, 73 decision bits should have zero direct loss
gradient (55 relation bits + 18 entity bits). This is a count of supervised
positions, not independent examples or new positives.

Run tools/run_alias_supervision_preflight_20261020.sh from the pinned new
checkout. It hides GPUs, loads only local tokenizer/config files and datasets,
never trains or deploys, saves success/failure reports, and uploads through a
new Git branch with transfer retries. The old passing report cannot substitute
for checking the new entity masks. Local tests use SDK/tokenizer doubles;
actual new mask execution is pending the server report.

## Coverage is still insufficient for a new training claim

alias_review.json reports semantic coverage by positive edge, unit, head type and
source SHA. The repair training set has model_variable->obtained_by in 6 units
from 3 sources, and model_variable->sourced_from in only 2 units from 2 sources.
Three rain-data variable edges are one unit; three DEM aliases are one CHM unit.
No new source positives, independent validation sources or model results are
created by this revision. Internal validation still shares 9 source PDFs with
training. It has been used in development and is not a final unseen source test.

TRAINING_READY=false is intentional. Source coverage must be supplemented and
reviewed before preparing another paired seed42 training command. Changing
output format or masks does not justify reusing old dev scores, expanding to
three seeds, or weakening the original strict gate. Masked placeholder values
remain in teacher-forcing context; zero direct loss is not marginalization.
