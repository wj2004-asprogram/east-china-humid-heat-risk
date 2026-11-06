# Phase 3 reproduction and resource batches

> 公开版本说明：以下命令记录原本地归档流程，路径占位符不代表可直接执行的恢复命令。干净克隆请使用 [可移植复现入口](PROBABILITY_PUBLIC_REPRODUCE.md)。原 seal/清单保留为历史证据，公开脱敏文件使用独立 publication manifest。


Run in the isolated repository. The original data, original environment, phase-2
arrays and scientific configuration remain unchanged. Only existing 2000–2017
NPZ features are opened by the modelling commands. No calibration or final-test
year arrays are loaded.

Completed state (2026-11-05): all 58 predefined tree fits and 44 exported tree
models are complete, with no pending models or numerical failures. The strict
winner is 7 leaves, min_child_samples=500, reg_lambda=10, 928 rounds. Temporal
retains its separate 15/100/10, 667-round winner. All 32 supplementary LR fits
are also complete. No training process is left running by this task.

Three execution interruptions are retained as historical receipts. The final
interruption occurred after the last model had been saved; its recovery only
restored and exported that checkpoint, without refitting. The original remaining
invocation did not reach its final runtime/peak-memory receipt; individual fit
durations and checkpoints survive, while its aggregate runtime/peak are not
claimed as measured. Current registry and report supersede partial snapshots.

After setting `$python` and the environment as shown below, verify the completed
matrix without fitting:

```powershell
& $python -m tools.probability_phase3_verify
```

This checks all 44 model parameter/iteration bindings, model hashes and closed-year
metadata. The completed report generation also verifies the shared comparison
counts; the empty pending table retains five named columns. Final targeted tests
passed 16/16; log: `data/probability_phase3/validation-tests-completed.log`.

Historical snapshot (superseded by the 20:30 UTC resume): the run was interrupted around 2026-11-05 20:15 UTC. At that snapshot no long
training process remains active. Temporal has all 22 models; strict has one
completed search candidate and one interrupted candidate with no completed
checkpoint. The strict winner has not been selected. The authorized second
resource batch was not started. Preserve the interruption receipt; it is a
pending task, not a numerical failure. Register an actual continuation budget
before resuming after the original deadline. At 20:30 UTC strict search resumed within the unchanged first budget; see `data/probability_phase3/strict-resume-receipt-20261105T2030.json`. Completed checkpoints are reused;
the interrupted candidate must restart its fit.

```powershell
$env:PYTHONPATH="$PWD\.phase3_deps;$PWD\.phase2_deps"
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$python='<ERA5_SOURCE_ROOT>\.venv\Scripts\python.exe'
```

LightGBM 4.7.0 is isolated in `.phase3_deps`. Version and configuration hashes
are recorded in `data/probability_phase3/lightgbm/run1/execution_lock.json`.
Do not reinstall into the original environment.

The completed train-only resource pilot used:

```powershell
& $python -u -m tools.probability_phase3_lightgbm --pilot --output-dir data/probability_phase3/lightgbm/pilot100
```

This is one additional 100-round fit, excluded from the 58 formal fits. It has
no validation dataset and exports no probabilities. Existing output directories
must not be overwritten to repeat it; choose a new isolated directory.

The first formal batch uses exactly the eight temporal F4/D1 candidates:

```powershell
& $python -u -m tools.probability_phase3_lightgbm --stage search --scopes temporal --output-dir data/probability_phase3/lightgbm/run1
```

For an interrupted batch add `--resume`. A candidate with a saved model and
`trained.json` is restored and scored without fitting again. A completed result
is reused only when scientific configuration, input manifests and parameters
match. Selection is withheld until all eight candidates have terminal results;
pending candidates are not failed candidates.

The authorized follow-up resource batch is separate from the first budget. It
may start only after the first batch ends, with at least 20 GiB free on C:, no
conflicting external training task, and single-thread tree fitting. Its bound is
two hours, permitting the in-flight fit to finish. Once registered, pass
`--batch-plan data/probability_phase3/execution_batch2.json` to every continuation
command. This does not edit the original deadline or scientific configuration.

The resource decision is recorded separately in
`data/probability_phase3/execution_batches.json`. Its initial budget is three
hours from the first formal search launch. A fit in progress may finish; new
fits are refused after that deadline. This schedule does not change scientific
parameters or the 58-fit target. A later authorized batch needs its own recorded
deadline; do not silently reset the original budget.

After the complete temporal search, the frozen resource-priority command is:

```powershell
& $python -u -m tools.probability_phase3_lightgbm --stage core --scopes temporal --resume --output-dir data/probability_phase3/lightgbm/run1 --tasks 1:F4 1:F0 1:F1 2:F4 2:F0 2:F1 1:F0P 1:F1P 2:F0P 2:F1P
```

F4/D1 exports the selected search model without another fit. Other arms use its
fixed parameters and selected iteration count, without validation early stopping.
If resources remain, `--stage remaining --scopes temporal --resume` executes the
locked remaining-arm sequence. After all temporal arms finish, the resource
follow-up authorization permits strict-tree search if the original deadline
has not passed. Strict must complete its own eight-candidate grid to select a
winner; it cannot inherit the temporal winner. An incomplete grid stays pending.

The initial search source is preserved under `run1/source_snapshots`, with hashes
matching the original execution lock. Later scheduling changes record their
source hashes per invocation and do not invalidate the scientific fit bindings.

Generate the current report after any completed batch:

```powershell
& $python -m tools.probability_phase3_lightgbm_report
```

The report states completed, failed and pending counts. Exported models have
text model files, metadata, annual prediction NPZs, metrics and reliability
tables. `report` contains same-arm LR comparisons, the fixed phase-2 LR_F4
reference, predefined feature contrasts and pending model rows. Validation
results are already used for selection and are not independent test evidence.
Binary arrays are excluded from git; retain local arrays for reproduction.

Once modelling stops and the final scientific notes are written, refresh the
static figure's final input fingerprints and create the single overall manifest:

```powershell
& $python -m tools.probability_phase3_lightgbm_report
& $python -m tools.probability_phase3_moisture_figure
& $python -m tools.probability_phase3_feature_ladder_figure
& $python -m tools.probability_phase3_archive
```

Both figures export PNG, SVG, yearly values, mean values and source-hash metadata
under `figures/probability_phase3`. The archive hashes all phase-3 outputs,
including local arrays, but never copies arrays into git or a bundle. Run archive
only after final outputs and notes stop changing.

Targeted tests (use a fresh writable temporary directory):

```powershell
& $python -m pytest tests/test_probability_phase3_lightgbm.py tests/test_probability_phase3_lr_ablations.py --basetemp .test-tmp-phase3-final -q
```

The delivered closeout ran these 16 tests successfully, with its log saved as
`data/probability_phase3/validation-tests-closeout.log`. Registry/report/figure
refreshes do not resume fitting. To rebuild the registry from saved files only:

```powershell
& $python -c "import json; from pathlib import Path; from types import SimpleNamespace; from tools.probability_phase3_lightgbm import update_registry, refresh_tables; a=SimpleNamespace(output_dir=Path('data/probability_phase3/lightgbm/run1')); update_registry(a,json.loads(Path('configs/probability_phase3_run.json').read_text())); refresh_tables(a)"
```

For the eventual continuation, use the unchanged scientific configuration and
the separately recorded active budget, first `--stage search --scopes strict
--resume`; after all eight candidates finish, execute strict core and remaining
arms in the same predeclared order. Do not inherit the temporal winner.
