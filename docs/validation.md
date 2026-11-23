# Testing and compatibility

Tests use synthetic fixtures or explicitly supplied external resources. No private dataset, personal path, named environment, or checkpoint bundle is required for the normal test run.

## Core checks

Install the development extra and run:

~~~bash
python -m pip install -e ".[dev]"
python -m pytest -q
~~~

The core tests cover annotation semantics, audio preparation, persistence and updates, conversion, and quality selection with small synthetic fixtures.

When changing the Skill's runtime helper or source adapter, smoke-test the affected command with representative input. For runtime discovery, check both a working setup and a missing capability.

## Model checks

Provide compatible resources through MDS_RESOURCES and representative audio/transcript records through MDS_INTEGRATION_SAMPLES. See [test_inference.py](../tests/test_inference.py) for their shapes.

MDS_DEVICE overrides the device saved in `local/agent-setup.json`. Supply one of these device settings for inference. Integration outputs stay in the test runner's temporary directory. Private resource and fixture descriptions belong in an ignored directory.

Opt-in integration checks exercise real model composition, per-item preflight results, persistence, constrained and unconstrained notes, persistent workers, in-memory audio, and DataLoader spawning.

## Upstream defects

For a model regression, preserve the failing candidate and a reproduction with the input, constraint, and backend revision. Fix defects in the owning repository and validate the updated submodule revision here.

Exercise hand-edited and off-grid tiers through import, note extraction, partial update, TextGrid reload and training export. Check ordered nearest-boundary matching, local frame collisions and stable re-projection. Audio endpoint discrepancies produce warnings. Invalid source sample ranges and impossible interval geometry still identify the affected item.

## Consumer and workflow checks

Run a small end-to-end pilot with the actual source format, models, output storage, destination validator, and export target before processing a large collection. Check resume and partial updates against preserved layers and unselected samples.

