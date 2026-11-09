# Acquiring and preparing recordings

The installable package supports provider downloads and user-supplied recordings. The synthetic demo generates its observations at runtime and needs no external data. Installing the package and running the demo do not require the supplied `results/` directory.

Fresh runs create new model, calibration and prediction identities. They do not reproduce the archived checkpoints or the paper's exact outcome counts. The supplied results retain their own source bindings and reproduction instructions in [results documentation](RESULTS.md).

## Download or import recordings

```bash
# Public provider download, with no training or inference.
safeprocess fetch-data --datasets four_tank --data-root /path/to/recordings

# Import the HAI recording ZIP obtained from IEEE DataPort.
safeprocess fetch-data --datasets hai --hai-archive /path/to/hai-recordings.zip \
  --data-root /path/to/recordings

# Alternatively use the direct HTTPS archive URL provided by the dataset host.
safeprocess fetch-data --datasets hai --hai-url "$HAI_ARCHIVE_URL" \
  --data-root /path/to/recordings
```

A new `run` or `describe` command automatically attempts missing-input acquisition before preparing or training any model. The same `--hai-archive` and `--hai-url` options can be passed directly to either command. `--no-download` uses existing recordings only. Preflight is read-only and never downloads. A frozen run resumes from its bound prepared inputs without reacquiring source data.

FourTank downloads the approximately 30 MB ZIP directly from the [provider repository](https://github.com/thomasbierweiler/FaultsOf4-TankBatchProcess), pinned to commit `64f3904dc101713062b9f8d0ecdf787d17bed702`. The downloader verifies the Git blob identity, extracts the measurement CSV and records SHA256 identities. Existing files are reused without replacement.

The [HAI provider repository](https://github.com/j-ehrhardt/hai-cpps-benchmark) contains simulation code. Its [download documentation](https://j-ehrhardt.github.io/hai-cpps-benchmark/datasets.html) directs recordings to [IEEE DataPort](https://ieee-dataport.org/open-access/hai-cpps-hamburg-ai-benchmark-cyber-physical-production-systems-v2). A stable anonymous direct archive endpoint could not be verified. The importer therefore accepts the actual provider archive or a direct HTTPS URL supplied by the user. Missing HAI data produces an actionable acquisition receipt, rather than substituting a different dataset.

Select the recording export with `continuous-normal` and `continuous-anomalies` CSV directories and adjacent `sim_setup.json` files. The current simulation repository describes a different export layout; its current branch must not be silently regenerated as a substitute. No release number is assigned to the paper export where the retained acquisition metadata does not establish one. The importer preserves source files, extracts only selected continuous DS configurations and validates the required archive layout. The dataset adapter checks channel and operating-context schemas during preflight or preparation. Existing files are retained, and conflicting contents cause an error.

SWaT is never downloaded automatically. Acquire its physical recordings under the provider's access procedure and place them in the layout below. No credentials are requested or stored. The `acquisition.json` receipt records statuses and hashes; URL query credentials are excluded. Recordings, archives and generated runs stay outside Git.

## Providers and permissions

| Dataset | Acquisition source | Required material |
| --- | --- | --- |
| HAI-CPPS DS1–DS10 | [Benchmark description](https://j-ehrhardt.github.io/hai-cpps-benchmark/), [provider repository](https://github.com/j-ehrhardt/hai-cpps-benchmark), dataset DOI `10.21227/5ewb-cn40` | Continuous normal and anomalous CSVs, with each scenario's `sim_setup.json` |
| FourTank | [Bierweiler/Labisch provider repository](https://github.com/thomasbierweiler/FaultsOf4-TankBatchProcess) | `SmA-Four-Tank-Batch-Process_V2.csv`, obtained under the provider's terms |
| SWaT | SUTD iTrust dataset provider, under its access terms | Physical normal and attack workbooks, plus `List_of_attacks_Final.xlsx` when available |

HAI here means **HAI-CPPS**, not the separate HAI ICS security dataset. FourTank measurements and SWaT recordings are externally acquired data. The repository license does not grant rights to redistribute them. The author approved retaining the short FourTank observation excerpts on 6 November 2026, with source attribution and notices recorded in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

No network capture, private object-storage account, ontology server or external reasoner is needed. The FourTank description PDF supplies engineering provenance for the bundled authored mapping but is not a runtime dependency.

## Directory layout

`--data-root` can identify either the parent collection below or the root of an individually selected dataset. Use portable filenames and preserve the provider's scenario directory names.

```text
recordings/
├── HAI/
│   └── Datasets/
│       ├── continuous-normal/
│       │   ├── ds1/
│       │   │   ├── <normal-recording>.csv
│       │   │   └── sim_setup.json
│       │   ├── ...
│       │   └── ds10/
│       │       ├── <normal-recording>.csv
│       │       └── sim_setup.json
│       └── continuous-anomalies/
│           └── ds<number>_<component>_anom_<condition>/
│               ├── <scenario-recording>.csv
│               └── sim_setup.json
├── FourTank/
│   └── raw/
│       └── SmA-Four-Tank-Batch-Process_V2.csv
└── SWaT/
    └── raw/
        ├── List_of_attacks_Final.xlsx
        └── Physical/
            ├── SWaT_Dataset_Normal_v1.xlsx
            └── SWaT_Dataset_Attack_v0.xlsx
```

For HAI, the `Datasets/` level can also be the selected dataset root. The adapter reads only `continuous-normal` and `continuous-anomalies`. It requires one continuous normal CSV for each selected DS configuration. Every selected CSV needs a neighboring `sim_setup.json`.

The FourTank CSV may also sit directly in `FourTank/`. Its delimiter is `;`. The original column names, including spaces and `ValueY`, must be preserved.

For SWaT, `.xlsx` and `.xlsm` physical workbooks are supported. A user-supplied Parquet equivalent with the same normalized schema is also accepted, using the same basename and `.parquet` extension. Accepted physical locations are the SWaT root, `Physical/`, `raw/Physical/`, `raw/` and `cache/physical/`. Parquet is preferred when both representations exist.

An optional SWaT attack-table transcription may be supplied as `attack_metadata.csv` or `physical_attack_metadata.csv` at the SWaT root. It needs `attack_number`, `start` and `end` columns. The original provider workbook is preferred. Missing metadata leaves attack numbers unknown where they cannot be reconciled with the label runs. It does not create a process-response qualification.

## Configurations and input roles

The paper preset contains these twelve model configurations. The ordered sensor names and owners are distributed in [paper_channels.json](../src/safeprocess/resources/paper_channels.json).

| Configuration | AE sensor channels | Observable operating context |
| --- | ---: | --- |
| `hai_ds1` | 11 | HAI component state indicators |
| `hai_ds2` | 16 | HAI component state indicators |
| `hai_ds3` | 8 | HAI component state indicators |
| `hai_ds4` | 8 | HAI component state indicators |
| `hai_ds5` | 19 | HAI component state indicators |
| `hai_ds6` | 32 | HAI component state indicators |
| `hai_ds7` | 16 | HAI component state indicators |
| `hai_ds8` | 27 | HAI component state indicators |
| `hai_ds9` | 24 | HAI component state indicators |
| `hai_ds10` | 46 | HAI component state indicators |
| `four_tank` | 13 | `CuStepNo ValueY` |
| `swat` | 11 | `MV301`, `MV302`, `P301`, `P302`, `P401`, `P402` |

HAI context is taken from observable columns matching `<component>.state_<state>.active`. The causal context code checks binary values and exactly one active state per component. Commands, direct fault proxies and simulation parameters do not become AE sensor inputs.

The SWaT sensor order is `DPIT301`, `FIT101`, `FIT201`, `FIT301`, `FIT401`, `FIT501`, `FIT502`, `FIT601`, `LIT101`, `LIT301`, `LIT401`. The six context inputs remain separate. Timestamp, `Label`, `Normal/Attack`, FourTank deviation IDs and other annotations are never model or context features.

## Normal partitions and boundaries

All split coordinates are zero-based, half-open intervals `[start, end)` within their prepared recording. Window construction keeps segments separate. It never concatenates across recording, batch, exclusion or split barriers. The fresh training recipe uses length-60 windows with stride 10 and fits its min–max scaler on training values only.

| Dataset | Normal partition rule | Boundary handling |
| --- | --- | --- |
| HAI | Chronological 60% training, 15% validation, 10% calibration, 15% normal test | A 59-row gap is removed from the beginning of every partition after training |
| FourTank | Chronological whole normal batches: 21 training, 5 validation, 4 calibration, 5 normal test | No window crosses a batch start, deviation change, timestamp gap or excluded controller step |
| SWaT | After startup exclusion, chronological 55% training, 15% validation, 15% calibration, 15% normal test | A 59-row gap is removed from the beginning of every partition after training |

For chronological partitions, the adapter first calculates cumulative boundaries by flooring the cumulative fraction of available rows. It then advances each boundary after training by 59 rows. The preceding partition keeps its original end. Each resulting segment must contain at least 60 rows.

For FourTank, deviation ID 1 denotes normal operation. Undocumented ID 0 is excluded. The retained active controller-step sequence is `1, 7, 8, 3`. Step 2, rows before a batch start and incomplete normal batches are excluded. The paper recipe requires 35 complete normal batches. Any fragments of a retained batch keep that batch's split, and a timestamp gap still separates their window construction. No sensor is synthesized for the PDF-listed channel absent from the CSV.

For SWaT, `Normal_v0` and `Normal_v1` are versions of the same recording. They are explicitly recorded as aliases, never independent sources. The adapter prefers `Normal_v1` and excludes another **19,800 rows**, accounting for the 1,800 startup rows already removed from that export. If only `Normal_v0` is supplied, it excludes 21,600 rows. The manifest records the selected export, source hash and trim. Startup rows remain outside every normal partition.

Fault recordings cannot enter any training, validation, normal-calibration or normal-test segment. Development and assessment scenarios are selected downstream from the declared source roles. See [WORKFLOW.md](WORKFLOW.md) for the full ordered workflow.

## Native annotations and development roles

The fresh protocol retains source-native perturbation descriptions separately from model inputs. It does not infer a physical process response from the residuals or from the provider's fault name.

- For HAI, scenario names ending in `pump50` supply `decision_fit`, those ending in `pump75` supply `risk_calibration`, and other scenarios supply `assessment`. If byte-identical native exports imply conflicting roles, all of those aliases remain assessment and share a source identity.
- For FourTank, chronological whole batches cycle through `decision_fit`, `risk_calibration` and `assessment` within each documented deviation. Every fragment of a batch retains its batch role.
- For SWaT, chronological disjoint attack-label episodes cycle through the same three roles. Segment-level roles override the recording-level assessment default. Source attack numbers are attached when metadata overlaps an episode unambiguously. A source label-run index is retained separately.

Real native records default to `native_category=locally_unverified` and `response_qualification=locally_unverified`. A command, attack annotation or deviation ID does not establish measured response support. The demo alone supplies explicitly synthetic native categories, with `synthetic=true` and `response_qualification=synthetic_only`.

These are versioned fresh-run membership rules. The original reported study retains its own closed memberships and qualification evidence.

## Preflight and preparation

Check the acquired schemas and FourTank boundaries before preparing recordings:

```bash
safeprocess describe --config paper \
  --data-root /path/to/recordings \
  --work-dir /path/to/new-run \
  --preflight
```

Use `--datasets hai_ds1 swat` to select configurations. Remove `--preflight` to write portable recording Parquet files, the acquisition manifest and the RDF description. These commands do not train a model. Fresh model execution belongs to the subsequent workflow stages.

The implementation checks on 6 November 2026 passed all twelve local schema preflights. They confirmed FourTank's 21/5/4/5 normal-batch split. **SWaT's local preflight inspected physical Parquet equivalents**, not a newly converted workbook campaign. A small synthetic workbook test separately exercised header normalization and timestamp/attack-table parsing. No complete real-data training or inference campaign was part of that validation.

Prepared recording paths are relative to the generated manifest. SHA-256 identities and source-row coordinates are retained. Moving the complete prepared data directory preserves loading without the original acquisition root. Keep recordings, weights, credentials and generated run outputs outside Git.
