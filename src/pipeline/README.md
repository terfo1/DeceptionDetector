# Pipeline Runner

This module runs existing project stages in one command. It does not create new preprocessing logic, model architectures, or training code; it only executes the existing modules and records which stages passed, failed, or were skipped.

Full pipeline:

```bash
python -m src.pipeline.run_pipeline --mode full
```

Preprocessing only:

```bash
python -m src.pipeline.run_pipeline --mode preprocess
```

Training only:

```bash
python -m src.pipeline.run_pipeline --mode training
```

Reports only:

```bash
python -m src.pipeline.run_pipeline --mode reports
```

Dry run:

```bash
python -m src.pipeline.run_pipeline --mode full --dry-run
```

Supported modes:

- `validate`
- `preprocess`
- `split`
- `baselines`
- `sequences`
- `neural`
- `training`
- `reports`
- `selection`
- `tracking`
- `full`

Outputs:

- `reports/pipeline/pipeline_run_report.txt`
- `reports/pipeline/pipeline_run_summary.md`
- `reports/pipeline/pipeline_steps.csv`
- `reports/pipeline/latest_pipeline_run.json`
