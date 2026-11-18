# Working with the dataset-creator Agent

Tell the Agent what data to use, which source fields to trust, and what to deliver. The Agent inspects the data, presents evidence for unresolved decisions, and runs the operations in scope. Scripts execute the selected operations and return their per-item results. You can request any stage on its own.

| Stage | What you may need to decide | What the Agent hands back |
| --- | --- | --- |
| First setup or update | Environment and device, submodule action, or whether to install or download a required resource | Capability check and reusable local setup |
| Source investigation | Recutting, source fields to reuse, and any requested changes to names, speaker labels, or layout | Source mapping, capability gaps, and affected inputs |
| Song and lyric discovery | Whether lookup is in scope and which supplied sources may be used | Song identity, verified lyric source, and unresolved recordings |
| Slicing | Whether to keep existing cuts or create new ones | Clips and Audition `.csv` markers that record cut ranges on the original recording |
| Lyric preparation | How to handle missing lyrics | Paired clip transcripts or an editable lyric matching table |
| Forced alignment | Unresolved lyric or pronunciation choices that affect the input | FA annotations and per-item outcomes |
| Notes | Whether to accept independent notes when usable syllable boundaries are unavailable | Requested annotations with their actual layers |
| Quality | Whether to apply a retention rule or request copies for a defined review | Overview and actual selection when requested |
| Delivery | Formats, destination for each format, and audio inclusion | Maintained clip/lyric pairs and editable annotations, requested training derivatives, Agent decisions, and pending review items |

The [README](../README.md#annotate-a-dataset-in-one-run) includes copyable prompts for a complete task and individual stages. Outputs in a shared destination follow the [numbered delivery layout](workflow.md#delivery-layout).
