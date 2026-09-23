# Remove Shadow Free Train


This repository is created for 2110433, you need to download all of this as the first step

## Baseline Model download

```
./third_party/download_models.sh bmnet          # BMNet only (2 MB, just a clone)
./third_party/download_models.sh shadowformer   # ShadowFormer only (137 MB)
```

## Dataset Download
```
./data/download_data.sh      # full dataset (~2.6 GB)
```
## What baselines we have
```
python baseline/run_baseline.py --list
```

## Run baseline
```
python baseline/run_baseline.py --no-oracles                     # skip the GT ceilings
python baseline/run_baseline.py --only boundary_gain             # one method
python baseline/run_baseline.py --limit 20                       # quick check
python baseline/run_baseline.py --supervised bmnet shadowformer  # add the AI models

```
