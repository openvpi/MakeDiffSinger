# Convert or reuse annotations

Conversion does not require a model session. Verify the [source fields](source-fields.md) and the target's requirements.

The built-in importer accepts canonical TextGrid and DiffSinger CSV. Source tier mappings rename fields but do not establish their meaning. Mark pronunciation groups verified only when the source supports that interpretation.

Use `import_annotations.py --audio-root` when the audio is in a separate tree, and repeat `--tier-map SOURCE=TARGET` for verified names. Use `--groups pronunciation` only for verified pronunciation groups. Supply hand-edited TextGrids directly. For unusual source timing conventions, the importer accepts `--boundary-tolerance-ms` and `--audio-warning-ms` as integer milliseconds. Both default to 10. See the [timing contract](../../../../docs/annotations.md#time).

A name mapping cannot distinguish repeated or unnamed TextGrid tiers. Use an available TextGrid reader in the source adapter, identify each tier by its position and verified contents, and pass explicitly named interval lists to import_intervals. Position alone does not establish semantics, and the original TextGrid need not be rewritten.

For other inputs, write a small task-local adapter. The [adapter example](../assets/import_adapter.py) maps source records with audio references and second-based intervals. Adapt its reader to the verified format.

Check:

- Sample identity and audio correspondence.
- Time unit, time origin, and endpoint coverage.
- Label semantics and missing layers.
- Which source fields can be represented, which remain available only in the source, and whether the requested conversion loses information.

Import partial annotations as partial. CSV `ph_num` describes declared syllables. A `words` layer requires evidence for pronunciation groups, and a `texts` layer requires the written lyrics.

Use the common validation and exporter. Compile only when the annotations supply the [complete training fields](../../../../docs/annotations.md#diffsinger-compilation). For CSV-only tasks, the exporter can omit audio explicitly. A training layout with WAVs needs available source audio.

If a conversion requires an unsupported or lossy interpretation, present the missing field or tradeoff. Rerun forced alignment or note extraction, shift boundaries, or change lyrics only within the authorized scope. Preserve missing scores as missing.
