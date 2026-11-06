# Life Research Resources — Cancer Subtype Classification from Genomic Variants

Classifying tumour type from a patient's gene-mutation profile: 26 cancer
subtypes predicted from variant calls across 4,384 genes.

| | |
| --- | --- |
| **Competition** | [2024 생명연구자원 AI활용 경진대회 : 인공지능 활용 부문](https://dacon.io/competitions/official/236355/overview/description) |
| **Host** | Ministry of Science and ICT · KRIBB · KOBIC · DACON |
| **Task** | Multi-class classification — 26 cancer subtypes (`SUBCLASS`) |
| **Data** | Tabular genomic variants, one column per gene (A2M … ZYX) |
| **Metric** | Macro F1 |

---

## The data shape

Each row is a patient. Each of the 4,384 columns is a gene, and the cell holds
either `WT` (wild type — no mutation) or a variant string such as `NANTAG>MFS`.

The matrix is overwhelmingly `WT`. Most of the work is turning a sparse grid of
short, irregular mutation strings into something a model can use.

## Preprocessing

### Pruning

Genes that are `WT` for every patient carry zero information and are dropped
outright. Patients who are `WT` across every gene are examined separately — their
subclass distribution is compared against the rest
(`wt_all_ratio / wt_not_all_ratio`), which shows certain subtypes (CESC, PCPG,
LAML, THCA, THYM) are over-represented among all-WT patients. That is signal in
itself: the *absence* of detected mutations is predictive, so these rows are kept
rather than discarded.

Missing values are filled with `WT` — a gene with no call is, for this purpose,
an unmutated gene.

### Normalising variant strings

Raw variant strings mix several conventions, and position indices appear
inconsistently. Positions are extracted and analysed for frequency first, then
stripped, so that `A123T` and `A456T` collapse to the same amino-acid
substitution. Entries that begin with `>` are malformed rather than meaningful
and are removed.

### Variant embedding

Each normalised variant is turned into a fixed-length integer sequence:

1. **Split on `>`** into the before and after parts —
   `'NANTAG>MFS'` → `['NANTAG', 'MFS']`
2. **Integer-encode** each amino acid and special token:

   ```
   X,* → 1   fs → 2   del → 3   wt → 4   - → 5   > → 6   sv → 7
   A → 8   C → 9   D → 10  …  Y → 27   (padding) → 0
   ```

3. **Compute the maximum length** across all variant sequences, separately for
   the before and after parts
4. **Right-pad with zeros** to that length, giving every variant a fixed shape
5. **Embed** the padded sequences through an embedding layer into dense vectors

Treating a variant as a short *sequence* rather than a categorical label is what
lets the model generalise across mutations it has never seen — chemically similar
substitutions land near each other in the embedding space.

## Modelling

Padded sequences for all genes are concatenated into one tensor per patient
(assembled in parts and `torch.cat`-ed to stay within memory), then fed to a
neural classifier over the 26 label-encoded subtypes.

Because the subtype distribution is heavily imbalanced,
**`iterative-stratification`** provides multi-label-aware stratified folds, and
**`torcheval`** computes macro F1 during training — the metric that is actually
scored, rather than accuracy, which a majority-class predictor would win.

Gradient-boosting baselines (XGBoost, CatBoost, LightGBM) are kept alongside for
comparison.

## Setup

```bash
git clone https://github.com/paulms77/Life-Research-Resources.git
cd Life-Research-Resources

pip install torch torcheval iterative-stratification \
            scikit-learn pandas numpy xgboost catboost lightgbm
```

Download the data from the
[competition page](https://dacon.io/competitions/official/236355/data) and set
the directory at the top of each notebook:

```python
dataset_dir = '/content/drive/MyDrive/life/'   # or '/kaggle/input/lifedatabase/'
```

## Usage

```
1. preprocessing.ipynb   EDA, WT pruning, variant normalisation, integer encoding
2. modeling.ipynb        embedding, padded-sequence assembly, training, submission
```

Run them in that order — `modeling.ipynb` consumes the tensors
(`train_padded_sequences_full.pt`) that `preprocessing.ipynb` produces.

## Notes

**Macro F1 punishes ignoring rare subtypes.** With 26 classes at very uneven
frequencies, a model optimised for accuracy collapses onto the common cancers and
scores badly. Stratified folds and class-aware evaluation matter more here than
model capacity.
