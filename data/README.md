# Data Directory

This project does not download datasets automatically.

## Required Dataset

- Dataset: PaySim fraud detection dataset
- Expected filename: `paysim.csv`
- Expected location: `data/raw/paysim.csv`

## Manual Setup

1. Download the PaySim dataset from your chosen source.
2. Rename the file to `paysim.csv` if needed.
3. Place it in `data/raw/`.

## Important Notes

- Do not commit the raw dataset to Git.
- Do not place processed outputs in `data/raw/`.
- Use the dataset validation utility before any future preprocessing step.

## Expected Required Columns

The current validation utility expects these columns:

- `step`
- `type`
- `amount`
- `nameOrig`
- `oldbalanceOrg`
- `newbalanceOrig`
- `nameDest`
- `oldbalanceDest`
- `newbalanceDest`
- `isFraud`
- `isFlaggedFraud`

Validation logic lives in [utils/dataset_validation.py](../utils/dataset_validation.py).
