# Context-Aware Supervision for Selective Sensor Reconstruction

Code, numerical results and supplementary material for the SafeProcess paper by Björn Ludwig and Oliver Niggemann, Helmut-Schmidt University.

The implementation compares sensor-reconstruction candidates from a fixed autoencoder and uses operating history to decide when to propose an estimate, abstain or retain concern.

## Run the code

```bash
pip install -e ".[dev]"
safeprocess demo --work-dir /tmp/safeprocess-demo

safeprocess run --config paper \
  --data-root /path/to/recordings \
  --work-dir /path/to/runs/fresh-paper --device cuda
```

The demo runs a small synthetic CPU workflow. The full workflow covers ontology descriptions, training, calibration, seven policies, fault/control scenarios, evaluation and the history ablation.
Fresh models receive new run identities and do not claim exact reproduction of the supplied results.

FourTank recordings can be downloaded automatically. HAI accepts a provider archive or direct download URL. SWaT requires provider access.
See [data acquisition](docs/DATA_ACQUISITION.md), [workflow and training](docs/WORKFLOW.md) and [ontology descriptions](docs/ONTOLOGY.md).

## Reproduce the reported results

```bash
pip install -r requirements.txt
python -B scripts/reproduce.py --out /tmp/safeprocess-results
```

This uses the supplied closed numerical evidence, without recordings, trained weights, fitting or model inference.
It regenerates policy comparisons, the history ladder, temporal and control summaries, figures and the matched example, and checks 31 observable decision fixtures.
A single `results/manifest.json` checks the result files' integrity.
Use `--verify-only` for just that check.

See the [result guide](docs/RESULTS.md), [method specification (MS-01–MS-10)](docs/METHOD.md) and [supplementary results](docs/supplementary-results.pdf).
The manuscript itself is maintained separately.

## Layout

- `src/safeprocess/`: installable implementation, ontology and authored mappings.
- `configs/`: fresh paper and synthetic-demo configurations.
- `results/`: closed numerical inputs, selected tables, trace excerpts and replay fixtures.
- `scripts/`: result reproduction and plotting.
- `docs/`: method, workflow, acquisition and supplementary results.
- `tests/`: bounded implementation checks.

Run directories, package builds and caches are generated locally and ignored by Git.
The code uses [Apache 2.0](LICENSE). Dataset excerpts and the CPSWatch schema have separate attribution and terms in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
