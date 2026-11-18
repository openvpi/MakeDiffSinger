# Dataset delivery

## Requested scope and adopted sources

Describe the selected data, source/license references, adopted fields, verified mappings,
and subset organization. Identify written lyrics, pronunciation readings, phoneme
sequences, timing or grouping, and notes separately where relevant. State which source
fields were used and which were regenerated. Include material choices on slicing,
missing lyrics, length outliers, and absent syllable constraints when applicable.

## Outputs and changes

Link the maintained clip/lyric pairs and editable annotations ahead of any training
derivative when the task prepares or annotates data. When slicing was done, link the
Audition cut-marker `.csv` files with the clips. Link any lyric matching tables with the lyric
outputs. For TextGrid delivery, name the actual tiers using the [annotation contract](../../../../docs/annotations.md#layers-and-identities).
State whether audio is included and point to companion state when the recipient will
maintain the annotations. Link the native alignment diagnosis and figures. Link pending
review items through the listening links in `05-review-and-decisions/review.md`. For applied quality exclusions, link `05-review-and-decisions/quality-review.md`.
Link `05-review-and-decisions/decisions.md` when the Agent applied a material decision,
including repairs that succeeded. Link recorded blocking issues and consequential processing exceptions.
Identify replaced layers and preserved samples or manual work.
Disclose any Git-tracked project files edited during this task and why.

## Counts and checks

Report source-recording, clip, and annotation duration separately where available.
State the relevant structural, conversion, and preservation checks.
Give completed, pending, failed, and excluded counts. For a quality filter, give
the affected IDs, reasons, rule, and denominator. Link the training CSVs for included rows.

## Pending work

List unresolved input decisions, skipped input errors, software failures, and the
concrete recovery step. Include any widespread anomaly awaiting input review.
