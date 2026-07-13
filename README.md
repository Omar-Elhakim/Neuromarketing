# Neuromarketing — EEG Emotion Recognition on SEED-IV

Deep-learning experiments that classify emotion from raw EEG, benchmarking nine
architectures on the SEED-IV dataset across subject-dependent and
subject-independent protocols.

## Overview

SEED-IV contains 62-channel EEG recorded from 15 subjects while they watched
72 film clips labelled with four emotions (neutral, sad, fear, happy). This
repository trains a range of models on it and compares them under two protocols
that answer very different questions:

- **Subject-dependent** — train and test on the same subject. Measures whether
  the emotion is decodable at all.
- **Subject-independent (LOSO)** — leave one subject out entirely, train on the
  other 14. Measures whether a model generalises to a new person, which is the
  only setting that matters for a deployable system. Accuracy drops sharply here,
  and that gap is the interesting result.

Tasks are run at 4 classes (the native labels), 3 classes, and 2 classes (positive
vs. negative).

Part of the motivation is consumer hardware: SEED-IV was recorded with a
research-grade 62-channel cap, but a neuromarketing product would use something
like an Emotiv EPOC+. Some of the DaViT experiments therefore subset SEED-IV
down to the 14 channels an EPOC+ actually has (`experiments/davit/dep_2class_*`
and `indep_4class.py`) to test whether the signal survives on affordable
hardware. `preprocessing/data25.ipynb` explores a separate 25-user dataset
recorded on a real 14-channel Emotiv device.

## Results

Accuracies below are read from the **recorded run output** in each notebook or
`*_output.txt`, averaged across subjects. `±` is the across-subject standard
deviation, reported where the run recorded one.

### Subject-dependent

| Model | Classes | Accuracy | File |
|---|---|---|---|
| BFENet | 2 (happy vs. sad+fear) | **96.26%** | `experiments/bfenet/dep_2class_happy_vs_sad_fear.ipynb` |
| DaViT | 2 | **94.86%** | `experiments/davit/dep_2class_v1.ipynb` |
| BFENet | 2 (+ neutral as negative) | 94.05% | `experiments/bfenet/dep_2class_happy_vs_sad_fear_neutral.ipynb` |
| DaViT | 2 (Emotiv EPOC+, 14 ch) | 93.32% | `experiments/davit/dep_2class_v2.py` |
| BFENet | 3 | **93.71%** | `experiments/bfenet/dep_3class.ipynb` |
| DaViT | 3 | 88.25% | `experiments/davit/dep_3class.ipynb` |
| DaViT | 4 | **85.02%** | `experiments/davit/dep_4class.ipynb` |
| DGCNN | 4 | 84.14% | `experiments/dgcnn/dep_4class_v4.ipynb` |
| RW-CCNN | 4 | 75.38% ± 11.45 | `experiments/rw_ccnn/dep_4class_v4.ipynb` |
| BFENet | 4 | 72.20% | `experiments/bfenet/dep_4class.ipynb` |
| Ro-CNN | 4 | 54.28% | `experiments/ro_cnn/dep_4class.ipynb` |
| CNN (baseline) | 4 | 42.30% | `experiments/cnn_baseline/dep_4class.ipynb` |

### Subject-independent (leave-one-subject-out)

| Model | Classes | Accuracy | File |
|---|---|---|---|
| DaViT | 2 | **83.41% ± 8.84** | `experiments/davit/indep_2class.py` |
| DaViT | 2 (no neutral) | 83.26% ± 11.79 | `experiments/davit/indep_2class_no_neutral.ipynb` |
| Attention-graph | 2 | 82.33% | `experiments/attention_graph/indep_2class_v2.ipynb` |
| DGCNN | 2 | 81.85% | `experiments/dgcnn/indep_2class.ipynb` |
| BFENet | 2 (+ neutral as negative) | 75.16% | `experiments/bfenet/indep_2class_happy_vs_sad_fear_neutral.ipynb` |
| DaViT | 3 | **64.13% ± 9.48** | `experiments/davit/indep_3class.py` |
| Attention-graph | 3 | 59.12% | `experiments/attention_graph/indep_3class.ipynb` |
| DaViT | 4 | **52.10% ± 8.44** | `experiments/davit/indep_4class.py` |
| Attention-graph | 4 | 34.95% | `experiments/attention_graph/indep_4class.ipynb` |

DaViT on differential-entropy features is the strongest model overall. The
subject-dependent → subject-independent drop on the 4-class task (85.02% →
52.10%) is the honest headline: cross-subject EEG emotion recognition is still
hard.


## Dataset

Download SEED-IV and place it at the repository root as `SEED-IV/`:

- <https://www.kaggle.com/datasets/phhasian0710/seed-iv>

The layout the code expects:

```
SEED-IV/
├── Channel Order.xlsx
├── eeg_raw_data/{1,2,3}/<subject>_<date>.mat   # 3 sessions x 15 subjects x 24 trials x 62 channels
└── eeg_feature_smooth/                         # precomputed DE / LDS features
```

Labels are fixed per session and identical across subjects, and each class has
18 trials, so the raw dataset is balanced. `preprocessing/data25.ipynb` uses a
different dataset ([25-user Emotiv recordings](https://www.kaggle.com/datasets/daviderusso7/seed-dataset)),
expected at `Data25/`.

## Preprocessing

`src/functions.py` implements the raw-EEG pipeline:

1. 4th-order Butterworth **bandpass, 4-50 Hz**
2. **Downsample** 1000 Hz -> 200 Hz
3. **Z-score** normalisation per channel
4. **Segment** into 4-second windows (800 samples) with 10% overlap

The feature-based experiments (DGCNN, DaViT, CCNN) skip this and read SEED-IV's
precomputed differential-entropy features with LDS smoothing from
`eeg_feature_smooth/` via TorchEEG.

The experiment notebooks were written to run on Kaggle — they install their own
dependencies and expect the dataset under `/kaggle/input/`. Each one is
self-contained.

## Project structure

```
src/                    # preprocessing library + runnable example
preprocessing/          # dataset exploration: SEED, SEED-IV, Data25
experiments/            # one folder per model family
├── davit/              # DaViT on DE+LDS features - strongest overall
├── bfenet/
├── dgcnn/
├── attention_graph/
├── rw_ccnn/  rw_cnn/   # raw-window CNNs
└── ccnn/  ro_cnn/  tsception/  googlenet/  cnn_baseline/
```

Experiment files are named `<protocol>_<n>class[_vN]`, where `dep` is
subject-dependent and `indep` is subject-independent. `_vN` marks successive
tuning iterations of the same configuration, in chronological order — the
highest `N` is the latest.

## Tech stack

Python, PyTorch, [TorchEEG](https://github.com/torcheeg/torcheeg),
PyTorch Geometric, scikit-learn, SciPy/NumPy, Plotly.

## License

[MIT](LICENSE)
