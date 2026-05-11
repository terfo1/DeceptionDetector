import pandas as pd

train = pd.read_csv("data/processed/train_windows.csv")
val = pd.read_csv("data/processed/validation_windows.csv")
test = pd.read_csv("data/processed/test_windows.csv")

train_users = set(train["participant_id"].dropna().unique())
val_users = set(val["participant_id"].dropna().unique())
test_users = set(test["participant_id"].dropna().unique())

print("Train participants:", train_users)
print("Validation participants:", val_users)
print("Test participants:", test_users)

print("Train ∩ Validation:", train_users & val_users)
print("Train ∩ Test:", train_users & test_users)
print("Validation ∩ Test:", val_users & test_users)

if not (train_users & val_users) and not (train_users & test_users) and not (val_users & test_users):
    print("OK: no participant leakage")
else:
    print("ERROR: participant leakage found")