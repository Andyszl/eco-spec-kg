# Training-source semantic evidence follow-up

The completed seed42 paired dev22 run did not meet its preregistered expanded
method/source path requirement (0/3). It is not eligible for three-seed expansion.
Results were received on branch `codex/seed42-pair-dev-20261015-100752-1631711`,
commit `97b112416ef0c70136e47d19acdda3e1a247c88e`.

This change repairs candidate evidence attachment only. Formula-package method
and source candidates retain their previous equation evidence and additionally
cite existing source spans actually mentioning their names. Their relations
also cite the head variable's definition. Candidate membership, names, types,
IDs, selection prompts and the frozen structural selector are unchanged in the
693 original training sources and 3 long views. No semantic relation is selected
automatically. Candidate generator version is v2.10 to distinguish the evidence
change from the evaluated v2.9 implementation.

Four new regression tests failed before the fix; those and the related candidate,
binding, scope and structural tests passed afterward (49 tests). The training
source comparison found evidence changes in 11 units, affecting 29 entities and
50 relations. These are local checks, not server model results.

The separate annotation file amends one training unit only. Its original
definition explicitly states that K_EPIC is calculated using the erosion-productivity
assessment model. It adds that named method and its obtained_by edge with the
definition span as evidence. The annotation is an assistant-reviewed derivative,
not new independent human gold. Frozen data and pilot training inputs remain in
their original versions. Source labels did not come from dev answers.

Actual repair-train supervision has two variable-method positives in two units
and two variable-source positives in one unit; internal validation has neither
role. The completed pair did not add positives in these roles. A focused review
of 43 training candidate occurrences retained four existing positives, found the
one explicit omission above, retained nine generic/boundary exclusions and
withheld 29 cases needing direct/derived source or method-scope adjudication.
Unlabeled candidates are not all known semantic negatives.

`training_ready=false`: training inputs have not been rebuilt for this annotation
version, and no new real-tokenizer preflight or GPU run has occurred. Existing
v2.9 training/evaluation launchers intentionally retain their original source and
protocol locks and must not be bypassed for this work. Rebuild and audit the new
targets, settle the remaining scope decisions, and register validation coverage
before issuing another fixed-seed paired training command. Preserve the original
full-dev gate and reserve an unused source for final independent validation.
