# Threshold Calibration Summary

- Best available model by current F1 if detectable: random_forest (test F1=0.7273)
- All-medium issue: test/causal_tcn, test/gru, test/lstm, validation/causal_tcn, validation/gru, validation/lstm
- Narrow probability range: test/causal_tcn, test/gru, test/lstm, validation/causal_tcn, validation/gru, validation/lstm
- Preliminary threshold warning: thresholds are diagnostic only until more participants are collected.
- Recommended next action: collect more data, retrain, then calibrate thresholds on validation data.
