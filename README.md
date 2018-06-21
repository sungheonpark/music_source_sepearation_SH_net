# Music source sepeartion using stacked hourglass networks
This is the code for the paper 'Music source separation using stacked hourglass networks', ISMIR 2018

## Usage

Required packages

```
tensorflow, pysoundfile, librosa, bss_eval (https://github.com/craffel/mir_eval)
```

## Dataset

[MIR-1K](https://sites.google.com/site/unvoicedsoundseparation/mir-1k) dataset

[DSD 100](https://github.com/faroit/dsdtools) dataset

## Training
Set the dataset and checkpoint paths at config.py and run

```
python train_mir_1k.py
```
for MIR-1K dataset, or

```
python train_dsd_100.py
```
for DSD 100 dataset.


## Evaluation
Run

```
python eval_mir_1k.py
```
for MIR-1K dataset, or

```
python eval_dsd_100.py
```
for DSD 100 dataset.
