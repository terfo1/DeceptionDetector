# Recommendations

## Data Collection
- Collect more participants before making scientific claims about model reliability.
- Keep the controlled experimental protocol and preserve participant identifiers for subject-independent splitting.

## Model Retraining
- Retrain all models from scratch after adding new participants.
- Fine-tuning can be added later, but full retraining is preferred for comparable experiments.

## Evaluation
- Compare F1, ROC-AUC, false positive rate, and false negative rate together.
- Treat single-class prediction behavior as a model failure mode.

## Threshold Tuning
- Consider threshold tuning only after a larger validation split is available.
- Keep false positive rate visible because truthful responses flagged as deceptive are ethically sensitive.

## Real-Time Readiness
- Run offline diagnostics before any real-time simulation.
- Use the Causal TCN as a candidate for future streaming compatibility, but do not deploy it from the current small dataset.

## Ethical Interpretation
- The system estimates deception risk under a controlled protocol.
- It must not be presented as a universal lie detector.

## How to update the models after collecting more data

For clean scientific evaluation, retrain from scratch after adding new participants:

```bash
python -m src.preprocessing.validate_raw_data
python -m src.preprocessing.build_windows
python -m src.training.create_splits
python -m src.models.train_baselines
python -m src.models.evaluate_baselines
python -m src.training.build_sequence_dataset
python -m src.models.train_sequence_models
python -m src.models.evaluate_sequence_models
python -m src.models.train_tcn_model
python -m src.models.evaluate_tcn_model
python -m src.analysis.generate_model_comparison
```

## Current Diagnostic Recommendations
- Collect more participants before treating model metrics as scientifically reliable.
- Keep the subject-independent split so participants do not appear in more than one split.
- After adding participants, retrain all models from scratch for comparable experiments.
- Check class balance in every split before interpreting precision, recall, and F1.
- Inspect prediction probability distributions, not only hard labels.
- Consider threshold tuning later, after a larger validation set is available.
- Keep false positive rate as a critical metric because truthful responses flagged as deceptive are ethically sensitive.
- Do not interpret current results as final reliability; they validate the technical pipeline only.
- Investigate prediction collapse for: causal_tcn, lstm.
- The LSTM collapsed to single-class behavior and should not be selected without retraining on more data.
- random_forest is currently best by available F1, but this should be treated as prototype-level evidence.
- The Causal TCN may need substantially more data before its temporal convolution capacity is useful.
- Current dataset size is below reliability thresholds; prioritize data collection over architecture changes.
