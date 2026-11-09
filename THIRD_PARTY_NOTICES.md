# Third-party material and data access

The root [LICENSE](LICENSE) applies Apache 2.0 to this repository's code. Dataset and third-party material retain their own terms and attribution.

## Observation excerpts

The closed results include derived residuals, candidate evidence, outcome tables and short waveform excerpts for the illustrative traces. They contain no complete recording or trained AE checkpoint.

The FourTank trace payloads in `results/traces/trace_*_inference.npz` include measured observations. Corresponding `trace_*_evaluator.npz` files include clean-reference excerpts. The useful example's raw and clean arrays each contain 300 rows and 13 channels. These are source-derived waveforms.

The retained acquisition record identifies Siemens material with an all-rights-reserved notice. The author approved inclusion of these short excerpts on 6 November 2026. Source attribution remains included.

## Datasets

- **HAI-CPPS:** Hamburg AI benchmark for cyber-physical production systems, configurations DS1–DS10. It differs from the HAI ICS security dataset. Provider links are in [DATA_ACQUISITION.md](docs/DATA_ACQUISITION.md).
- **FourTank:** Siemens Smart Automation four-tank batch measurements described by Bierweiler and Labisch. The acquisition guide identifies the pinned provider repository.
- **SWaT:** the eleven-channel subset described in the method specification. Original recordings require SUTD iTrust access.

Source identifiers and coordinates do not grant recording access or redistribution rights. Users obtain complete recordings under the provider's terms.

## CPSWatch ontology

`src/safeprocess/resources/CPSWatch-1.0.1.ttl` bundles CPSWatch 1.0.1 under Creative Commons Attribution 4.0 International. Its original author, version and license statements are preserved. See the [schema notice](src/safeprocess/resources/NOTICE.md). Authored channel and topology mappings identify their engineering sources and do not redistribute the provider recordings or FourTank description PDF.
