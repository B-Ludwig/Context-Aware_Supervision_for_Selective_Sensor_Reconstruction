# Validation and reproduction scope

The standalone implementation has bounded CPU checks for training/resume, TTO, source isolation, context/graph rules, seven-policy guards, evaluator separation and acquisition. The synthetic workflow exercises all stages with a deliberately small recipe. Its scores are not paper results.

All twelve real-data configurations were schema-preflighted, totaling 231 sensor channels. FourTank checks cover its 21/5/4/5 normal-batch split. HAI archive import was checked with actual DS1 exports. The direct HAI HTTPS path uses mocked provider responses because no stable anonymous DataPort archive URL was verified. SWaT requires user access.

The reported numerical comparisons are checked independently of fresh training by `scripts/reproduce.py`. This includes primary, history and temporal tables, witness/control accounting, supported-native graph outcomes, all 712 temporal observations per policy, 31 observable fixtures and the matched-case calculation.

The fresh training recipe, CUDA performance and scores from fresh full-data runs have not been validated by a complete scientific campaign. Fresh weights do not claim exact equality with the models behind the supplied results.

Run implementation checks with:

```bash
pip install -e ".[dev]"
pytest
```

Run closed reproduction with:

```bash
python -B scripts/reproduce.py --out /tmp/safeprocess-results
```

Recordings, trained weights and generated run/build directories are excluded from Git. Short FourTank trace excerpts are included with attribution as described in the third-party notices.
