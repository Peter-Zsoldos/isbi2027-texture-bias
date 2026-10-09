# ISBI 2027 submission 2: "Small regions look homogeneous: finite-sample bias of radiomic texture features and a model-free block-jackknife correction"

Paper source: `paper/main.tex` (build with `latexmk -pdf main.tex`). Data: MSD Task06 lung at `../data/msd/Task06_Lung` (see the paper-1 README for the download).

## Method
`corrections.py` — `evaluate(f, img, mask, k=2)` wraps **any** feature function `f(img, mask) -> dict` (for example the full PyRadiomics extractor). It returns the plug-in value and the delete-a-block jackknife over k^3 contiguous blocks (k=2: octants; unequal-block formula of Busing et al. 1999). It optionally also returns the split-half estimate.
`texfeat.py` — a fast NumPy re-implementation of the PyRadiomics first-order Entropy/Uniformity and GLCM JointEntropy, JointEnergy, Contrast, Idm and Correlation features (`validate.py` checks a 5-decimal match with PyRadiomics 3.0.1).

## Drop-in use (`radjk.py`)
```python
from radiomics import featureextractor
from radjk import extract_corrected
ext = featureextractor.RadiomicsFeatureExtractor(params)   # your usual configuration
feats = extract_corrected(ext, image, mask)                # adds jk_* corrected features, jk_xi, jk_neff, jk_reliable
```
About 1 s per tumour on RIDER (`rider/bench_radjk.py`).

## Reproduce
| result | command (Python 3.9 venv with pyradiomics for scripts that import it) |
|---|---|
| validation vs PyRadiomics | `python validate.py` |
| synthetic benchmark (Fig. 1a) | `python bench_synth.py 10 2 bench_full.csv` |
| real CT soft tissue, internal reference (Table 1, Fig. 1b) | `python real_regions.py ../data/msd/Task06_Lung 63 real_full.csv`; smoothed versions: `... 24 smooth_0.7.csv 0.7`, `... 24 smooth_1.4.csv 1.4`, `... 12 smooth_2.0.csv 2.0` |
| all 93 PyRadiomics features (Fig. 1c) | `python pyrad_wrapper.py ../data/msd/Task06_Lung 40 pyrad_full.csv 8`, then `python analyze_pyrad.py` |
| regime summary, gated estimator (Table 1, Fig. 1b) | `python analyze_regimes.py` (also writes `real_all_smooth.csv`) |
| bin anchoring (Fig. 1d) | `python anchoring.py` |
| lung tumours vs volume | `python tumour_volume.py ../data/msd/Task06_Lung tumour_full.csv 8`, then `python analyze_tumour.py` |
| figure | `python fig2.py` |

Early feasibility scripts and trial outputs: `pilots/`.

**Gated estimator** (recommended): use the jackknife value when N/xi^3 >= 20 (xi = 1/e autocorrelation length inside the ROI, `real_regions.corr_length`), otherwise keep the plug-in value and flag the ROI as unreliable.
