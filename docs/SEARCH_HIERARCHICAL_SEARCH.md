# Baby / Sports hierarchical CoLiftRec + Diffusion search

The search entry point is scripts/search_coliftrec_diffusion.py.

## Selection protocol

- MSCA is frozen at seed 999 and is not retrained by search trials.
- Validation users are split into SEARCH 80% / HOLDOUT 20% with seed 20262901.
- C1/C2/C3 search CoLiftRec; D1-D5 search Diffusion training and editing; J1 matches each Diffusion candidate to its own CoLiftRec baseline.
- HOLDOUT, crossfit, training-seed robustness and purification-seed robustness are used before a recommendation is frozen.
- Search itself must keep TEST_ACCESSED=false and TEST_USED_FOR_SELECTION=false.

## Resume

Reuse the same --search-dir and add --resume. Completed CSV trial IDs are skipped and cached MSCA / semantic assets are reused.

## Logging

Master search logs are written under log/search/. Launcher logs are shell stdout/stderr only. Search progress is also persisted in search_state.json, progress_summary.json, CSV files and JSON shortlist files.

## One-time final Test

scripts/evaluate_search_recommendation_test.py is intentionally separate from the search runner.

It only accepts a non-smoke search that:
1. completed with a passed no-Test audit;
2. froze STABLE_DIFFUSION_UPGRADE_FOUND;
3. has an intact frozen MSCA checkpoint, archived search config and selected Diffusion checkpoint.

The Test evaluator writes to runs/search_test/<dataset>/<search_id>/, not into the Validation search tree. It reports raw MSCA / CoLiftRec / Full metrics only; it does not compute Delta vs Test, does not alter selection, and refuses a second successful Test evaluation.

If no stable Validation recommendation is found, --skip-if-no-stable writes a skip status without accessing Test.
