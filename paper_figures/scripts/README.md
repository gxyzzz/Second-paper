# Figure scripts

- `extract_existing_data.py`: read frozen evidence and generate compact CSV files plus provenance.
- `validate_raw_data.py`: validate CSV integrity, seed policy, frozen M25 parameters and source resolvability.
- `collect_band_diagnostics.py`: TRAIN-only deterministic replay for Fig15/16; it never accesses Test.
- `make_all.py`: invoke plot scripts only; it never trains a model.
