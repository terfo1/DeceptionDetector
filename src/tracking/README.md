# Dataset Versioning and Run Tracking

This module tracks dataset versions and experiment runs using local JSON, CSV, and Markdown files. It does not use external experiment tracking services.

Run this after collecting new data and after rerunning the full machine-learning pipeline:

```bash
python -m src.tracking.generate_tracking_report
```

Outputs:

- `reports/tracking/dataset_versions.csv`
- `reports/tracking/experiment_runs.csv`
- `reports/tracking/latest_dataset_manifest.json`
- `reports/tracking/latest_run_manifest.json`
- `reports/tracking/tracking_report.txt`
- `reports/tracking/tracking_summary.md`
- `reports/tracking/reproducibility_checklist.md`
- `reports/tracking/file_hashes_latest.csv`
- `reports/tracking/metric_history.csv`

The manifests record raw and processed file hashes, dataset counts, available model metrics, the selected live model, and warnings that affect reproducibility.
