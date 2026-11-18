# Setup and recovery

## Use the saved runtime

Use the interpreter saved in ignored `local/agent-setup.json` unless the user specifies a different one. Read `configs/resources.local.json` for model and dictionary references. Check only the capabilities required by the current stage.

Inspection and conversion need only the core. Check the resources for the selected stage. When lyric matching requires new ASR hypotheses, check for `qwen_asr` and the chosen local model bundle.

### Select the inference device

Use the device the user specifies for the run. Otherwise use the saved `device` from `local/agent-setup.json` even if another local device appears faster or more suitable. Apply the same choice to all inference stages. The CLI accepts `--device cpu`, `--device cuda`, or an exact CUDA index such as `--device cuda:1`. Ask for a device only when an inference stage needs one and neither source specifies it.

If the model reports that its selected device cannot execute its kernels or precision, the affected model retries on CPU and records the reason. A missing device, memory shortage, or unrelated inference error follows ordinary recovery. Do not change the saved configuration or move other stages when one model falls back.

For forced alignment or note extraction under memory pressure, try a smaller `--batch-size` first. For more frequent on-disk checkpoints in `run_pipeline.py`, reduce `--chunk-size`, which controls how many samples are processed before saving. For throughput tuning, compare `--batch-size` and `--num-workers` on the same inputs. `--precision` and CUDA `--allow-tf32` change numeric execution. Record them with `--seed` when comparing results.

## Set up or repair a required capability

1. Inspect the relevant submodules. Report missing initialization, revision differences, or local changes and let the user choose the action. For an authorized initial checkout, use the [installation commands](../../../../docs/installation.md).
2. Ask for the environment name if there is no saved choice, then resolve its interpreter. If the environment needs to be created or repaired, use the same [installation guide](../../../../docs/installation.md) and the user's authorization. Verify the selected device with the chosen interpreter and report a failed check.
3. For a missing model, recommend an upstream bundle compatible with the stage and language. Give its source, requirements, approximate download size when available, and destination under `local/models/`. Keep the bundle's original folder name. Ask before downloading unless the task already authorizes the download.
4. Run the runtime helper with the chosen interpreter. Replace the placeholders with the agreed paths and device. Use `--require core`, `tifa`, `game`, or `all` for the capability being checked. Add `--check-device --device DEVICE` for compute verification. Then check G2P or model compatibility with a small relevant pilot when needed.

~~~bash
python .agents/skills/dataset-creator/scripts/resolve_runtime.py --tool-root . --require tifa --resources configs/resources.local.json --check-device --device DEVICE
~~~

After setup or an agreed update, add `--remember --environment ENV_NAME` to save `environment`, `python`, and `device` to the ignored setup file. Choose the check scope for the current task. Keep model and dictionary paths in the resource configuration.

## Classify failures before retrying

Use the stage's per-item diagnostics to locate the failure. When several clips show a similar symptom, test the suspected shared cause on an affected clip. Where the scope is unclear, also check a clip that uses another recording, lyric reference or resource.

| Evidence | Response |
| --- | --- |
| A required model, decoder, dictionary or other shared resource is missing or reproducibly fails | Hold the stage for clips that require that resource. Continue other languages, resources and independent stages. Resume the affected clips when the resource works. |
| Isolated invalid input, missing transcript, or unsupported pronunciation | Keep the item diagnostic and continue with valid items. For a G2P failure, [diagnose, validate, and apply straightforward repairs](pronunciation.md) before preparing the review list. Leave unresolved items pending or skip them and report the reason. |
| Isolated software failure on valid input | Record the failure, skip the item, continue other valid items, and report the defect. |
| Evidence ties an annotation defect to a shared input, such as shifted cut coordinates or a lyric sheet mapped to the wrong recording | Hold only the dependent clips and stages. Report the shared input, observed defect and affected IDs or source group. Correct it within scope and retry. If the user must supply a correction, identify exactly what they need to check and finish all unaffected work. A dataset-wide pause requires evidence that the failing dependency covers the entire dataset. |
| Several clips have homophones, omitted words, mixed scripts or uncertain readings | Keep first-pass FA and annotation moving on every usable candidate. Compare alternatives on the affected passages using the [lyric handoff](match-lyrics.md#hand-candidates-to-forced-alignment). Similar text discrepancies across songs alone do not establish a common failed dependency. |
| A song reference is unavailable or unreliable for this version or vocal part | Continue that group with independent clip ASR where available. Keep the other songs on their own verified references. |

If a comparison disproves the suspected shared cause, release the group and address the remaining defects per clip. For model-boundary failures, keep the candidate and compare it with the [time and layer contract](../../../../docs/annotations.md#time). For destination validation failures, compare the validator's expected semantics with the requested output before reporting the disagreement.

## Resume completed work

Use the [recovery commands](../../../../docs/workflow.md#recovery). Before reusing a stage, check the source identity, sample range, transcript, resource options, and edited or stale layers. Reuse current FA when only notes failed. Select the failed or missing items explicitly and protect newer edited assets when a problem record contains an older candidate.
