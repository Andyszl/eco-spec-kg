# Final blind material preparation - 2026-10-21

This branch contains raw official PDFs, fixed small-case passage scopes, ontology reference and blank human-review ledgers. It contains no new model predictions or gold labels. The Python tool only verifies and copies packet/. No GPU, Swift, or additional Python package is required on the receiving machine.

Run: `python tools/prepare_review_packet_20261021.py --out /new/absolute/review/directory`

The destination must be new and outside this immutable checkout. Successful integrity checking prints BLIND_PACKET_OK=true, MODEL_CALLS=0 and INFERENCE_READY=false. This does not approve source independence, human annotation or final inference. Human-review instructions are in packet/README.md.

The inference pipeline remains frozen at e69048ac27565f52fe8b06326c7530312de683a0. Old datasets and results are not included or changed. This is a material handoff, not a new training experiment, final evaluation launcher, or new score result.
