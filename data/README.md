# Data

Nothing here is needed to run the pipeline: by default it uses the synthetic
drifting stream from `src/data/synthetic.py`. Real CSVs go in `data/raw/`
(git-ignored).

## Dataset A: ULB Credit Card Fraud
- https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- 284,807 transactions, 492 fraud (0.172%), columns `Time, V1..V28, Amount, Class`.
- Put `creditcard.csv` in `data/raw/`, or:
  ```bash
  pip install kaggle   # kaggle.json token in ~/.kaggle/
  kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip
  ```

## Dataset B: IEEE-CIS Fraud Detection
- https://www.kaggle.com/competitions/ieee-fraud-detection/data
- Accept the competition rules on the page first, then:
  ```bash
  kaggle competitions download -c ieee-fraud-detection -p data/raw --unzip
  ```
- Use `train_transaction.csv`. The loader renames `isFraud` to `Class` and
  `TransactionDT` to `Time`.

Never train on one dataset and test on the other: their feature spaces differ.
