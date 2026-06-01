# Known Limitations

- The dataset is small, so metrics are prototype-level.
- `random_forest` is selected for the current prototype because it performs best on available test F1.
- `gru` is retained as fallback.
- `lstm` is disabled because prediction collapse was detected.
- `causal_tcn` is not reliable yet because its current probability range is narrow.
- There is no real eye tracker SDK integration yet.
- There is no production security, authentication, database, or Docker packaging.
- The system is not a universal lie detector and must not be used as an autonomous truth/lie decision tool.
