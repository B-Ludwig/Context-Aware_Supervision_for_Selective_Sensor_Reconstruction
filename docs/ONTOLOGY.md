# Portable ontology descriptions

`safeprocess describe` writes a local CPSWatch description for the selected recording configuration. The description connects ordered sensor channels to component owners, identifies operating-context sources and records acquisition identities. Training and policy code use the validated manifest directly. No ontology server, network request or reasoner is required.

## Included vocabulary and authored mappings

The package bundles the unchanged [CPSWatch 1.0.1 schema](../src/safeprocess/resources/CPSWatch-1.0.1.ttl). Its ontology namespace is `https://w3id.org/CPSWatch#` and its version identifier is `https://w3id.org/CPSWatch/1.0.1`.

CPSWatch is authored by **Björn Ludwig**, ORCID `0009-0003-2548-8225`, and distributed under **Creative Commons Attribution 4.0 International**. The [bundled license notice](../src/safeprocess/resources/CPSWatch-LICENSE.txt), ontology attribution and [resource notice](../src/safeprocess/resources/NOTICE.md) are retained. The repository's Apache-2.0 license does not replace those terms.

The following authored resources supply the dataset bindings:

| Resource | Purpose |
| --- | --- |
| [paper_channels.json](../src/safeprocess/resources/paper_channels.json) | Ordered sensor names, their unique component owners and identity encoding for all twelve configurations |
| [four_tank_metadata.yaml](../src/safeprocess/resources/four_tank_metadata.yaml) | Source-transcribed component/channel descriptions, documented deviations and declared connections |
| [swat_topology.yaml](../src/safeprocess/resources/swat_topology.yaml) | Declared physical system connections and component descriptions, with per-assertion evidence |
| [swat_channel_metadata.yaml](../src/safeprocess/resources/swat_channel_metadata.yaml) | Channel descriptions, observable properties and engineering-unit provenance |

HAI components and connections are read from each scenario's released `sim_setup.json`. Its declared model edges are projected from port names to their explicit component endpoints. The adapter checks structural consistency with the configuration's normal recording.

These mappings contain no trained weights, normalization statistics or calibration profiles. They do not grant rights to redistribute source recordings. References to engineering documents identify evidence for the mapping. They are not additional runtime input paths.

## Description and manifest

Each prepared dataset directory contains:

```text
manifest.json
ontology.ttl
ontology-validation.json
recordings/
  <recording-id>.parquet
```

The manifest schema is `safeprocess-dataset-v1`. It records:

- The dataset identity and ordered `features` used by the AE.
- Separate `context_columns`, their declared bindings and channel descriptions.
- One aligned `owners` entry per AE feature, the complete component list and explicit component `connections`.
- Source and prepared-recording hashes, source aliases, recording identities and relative paths.
- Half-open source segments and their normal/native roles, plus evaluator-only native annotations.
- The ontology path, schema hash and generated description hash.

The RDF uses CPSWatch monitoring systems, monitoring sensors and `hasMonitoringSensor` ownership. Sensor order is explicit through the local benchmark extension's `featureIndex`. Observable properties use SOSA's `observes` predicate. Known engineering units use CPSWatch's `hasUnit`. A missing unit remains unspecified.

Operating-state inputs are represented separately as `ControlStateChannel` resources. Source annotations remain outside the AE feature and context lists. Local benchmark-extension terms describe the acquisition manifest and do not change the bundled CPSWatch schema.

## Sensor, context and trust contracts

Every model sensor has exactly one owner. The authored feature order remains fixed through data preparation, training, scaler export and model reload. The `training_segments` interface returns sensor arrays in that order and excludes context and annotations.

Context comes from declared observable sources:

- HAI uses component state indicators matching `<component>.state_<state>.active`. A valid state has binary indicators with exactly one active state per component.
- FourTank uses the controller step `CuStepNo ValueY`.
- SWaT uses `MV301`, `MV302`, `P301`, `P302`, `P401`, `P402`, separately from the eleven AE sensors.

The context implementation constructs histories using information available through the window end. A declaration or a syntactically valid state does not authenticate a physical signal. The policy's context-validity contract assumes trusted operating-state inputs. Invalid context is distinct from a valid state with insufficient reference support or an unseen state requiring a flagged pooled fallback.

`Label`, `Normal/Attack`, `DeviationID ValueY`, injected target identities, clean references and attack annotations are evaluator information. They cannot become model inputs or operating context. Native source categories and response qualifications are retained for reporting, not learned from the source description.

## Process-flag graph semantics

The process flag uses **only explicitly declared `CPSWatch.hasSystemConnection` edges**. The declaration may retain source-documented or explicitly authored connections with their evidence status. The package does not add edges from shared parents, component containment, a control-loop relation or an external reasoner.

For a set of persistent peak owners, connectivity is evaluated on the graph induced by those owners. A declared edge contributes only when both endpoints are peak owners. Edge direction is ignored for this connectivity test. At least two distinct owners are required.

For example, declarations `A → X` and `X → B` do not connect peaks on `A` and `B` when `X` is not a peak owner. Adding a shared hierarchy parent also cannot connect them. This preserves the paper's distinction between distributed evidence and a connected supported graph. Disconnected or incomplete declarations can leave the process flag false despite persistent evidence on several components.

The graph is an explicit support condition for withholding an otherwise eligible reconstruction proposal. It does not establish the physical cause of the anomaly.

## Validation and Python access

Preparation validates unique ordered sensors, separate context columns, one owner per sensor, declared connection endpoints, portable recording paths, segment bounds and the exclusion of native recordings from normal partitions. The generated Turtle is parsed locally as a syntax check. Its OWL import is descriptive and is never fetched. These checks are structural validation, not a proof of the physical completeness of the plant graph.

```python
from pathlib import Path
from safeprocess.data import describe_dataset, load_manifest, read_recording, training_segments

prepared = Path("/path/to/new-run/prepared/hai_ds1")
manifest = describe_dataset("hai_ds1", Path("/path/to/recordings"), prepared)
normal = next(row for row in manifest["recordings"] if row["normal"])
frame = read_recording(manifest, normal["id"], verify_hash=True)
train_arrays = training_segments(manifest, "train")

# After moving the complete prepared directory:
manifest = load_manifest(Path("/path/to/relocated/hai_ds1/manifest.json"))
```

`describe_dataset` and `load_manifest` return a dictionary with an in-memory base-directory hint. The hint is not written into `manifest.json`. Relative recording paths resolve from the manifest's current directory, so closed prepared-data loading does not depend on the original acquisition path.

The data/ontology tests cover relocation, input-role isolation, split gaps, FourTank batch barriers, SWaT parsing and source aliasing, and the exclusion of graph paths through nonpeak components. Their synthetic inputs are generated during testing. No source recording or trained model is required for those checks.
