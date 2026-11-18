# Python API for task-local adapters

Start with the [adapter example](../assets/import_adapter.py) for a foreign label format, or the [Sample, AudioSource, and Transcript example](../../../../docs/api.md#in-memory-input-and-exact-slices) for custom audio and lyrics. Use the [annotation contract](../../../../docs/annotations.md) when mapping source labels or time units.

The [Python API guide](../../../../docs/api.md) covers preparation, annotation, validation, and export. For adapter tasks that need lower-level entry points, use:

| Work | API |
| --- | --- |
| Inventory a source tree | `makediffsinger.api.inspect_dataset` |
| Archive accepted cut markers | `makediffsinger.slicing.write_audition_markers` |
| Hand generated lyrics to FA | `makediffsinger.lyric_handoff.write_candidate_table(samples, report["clips"], table_path)` for matching or ASR results, then apply the table. For file-backed clips passed as Sample objects, `alignment_inputs(samples, delivery)` in the same module recovers missing pairs from those candidates. |
| Load failed or pending items | `makediffsinger.storage.load_problems` |
| Finish pending review after a custom run | `makediffsinger.review.sync_pending_review(annotation_root, review_root)` |

