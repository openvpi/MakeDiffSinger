# Working in MakeDiffSinger

- Use [dataset-creator](.agents/skills/dataset-creator/SKILL.md) for dataset work. When a task needs to run project scripts or import the core, use the [saved setup and runtime instructions](.agents/skills/dataset-creator/references/setup-and-recovery.md#use-the-saved-runtime). Check only the capabilities required by the task.
- Keep machine-specific settings and task artifacts in ignored files or the user's workspace. Preserve source data during dataset work.
- Use this repository's workflow and data contracts for annotation, validation, and compilation. For local files outside the checkout, read only task-supplied inputs, configured resources needed by the current stage, or references explicitly named by the user. Do not inspect other checkouts, datasets, or directories to infer conventions.
- Treat `external/TIFA` and `external/GAME` as read-only during dataset work. Report defects for upstream resolution. Ask before changing submodule initialization or revisions, and keep local algorithm replacements out of dataset runs.
- Keep dataset-specific edits outside Git-tracked project files. Disclose any tracked changes made for an authorized task. Repository maintenance requests do not authorize commits or pushes by themselves.
