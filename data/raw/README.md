# Raw data location

Place the genuine PaySim CSV at:

```text
data/raw/paysim.csv
```

The file must contain these columns with their original PaySim names:

```text
step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,
oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud
```

From the repository root, prepare it with:

```bash
.venv/bin/python scripts/prepare_data.py
```

The raw CSV and all prepared CSVs are intentionally Git-ignored. Do not add or
redistribute the dataset through this repository. PaySim is simulator-generated
data and remains subject to its source publication/host terms; the repository's
MIT license covers project code only.

