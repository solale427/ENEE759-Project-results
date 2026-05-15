# tail-risk-motion-prediction

Unsupervised driving-style clustering on Argoverse 2 (AV2) and Waymo Open Motion Dataset (WOMD) trajectory features, with replications of two recent baselines (KDSC and TDBM) for comparison.

## Layout

```
notebooks/
  AV2/                        # AV2 clustering + GIF visualization
  Waymo/                      # WOMD clustering + GIF visualization
  KDSC/                       # KDSC baseline replication + GIFs
  TDBM/                       # TDBM baseline replication + GIFs
  Comparison/                 # Per-method stats, continuous-score comparison
  archive/                    # Legacy exploratory notebooks
src/tailrisk_mp/              # Library modules used by notebooks and extraction
scripts/                      # Feature-extraction and MTR-setup utilities
artifacts/                    # Per-stage outputs (parquets, figures, JSON)
  phase1/, phase1_waymo/      # Raw per-agent feature tables (input)
  clustering/, clustering_waymo/
  kdsc_replication/{av2,waymo}/
  tdbm_replication/{av2,waymo}/
  comparison*/                # Cross-method comparisons
report/                       # LaTeX source for the project report
third_party/                  # UniTraj, ScenarioNet
_archive/                     # Old artifacts / unused modules (safe to delete)
```

## Pipeline

1. **Feature extraction** — AV2/WOMD scenarios are read through ScenarioNet via UniTraj, and a per-agent feature table is written to `artifacts/phase1/tables/` (AV2) and `artifacts/phase1_waymo/tables/` (WOMD).

   ```bash
   # AV2 (on a GPU node)
   conda run -n tailrisk-mp-cu126 python scripts/run_difficulty_analysis.py --device cuda

   # WOMD (SLURM)
   sbatch scripts/run_waymo_extraction.sbatch
   ```

2. **Clustering (our method)** — open the notebook for the dataset and run it top-to-bottom.

   - `notebooks/AV2/driving_style_clustering.ipynb`
   - `notebooks/Waymo/driving_style_clustering_waymo.ipynb`

   Outputs go to `artifacts/clustering/` and `artifacts/clustering_waymo/` respectively (parquets with `style_label`, `kmeans_K2.pkl`, cleaning reports, figures).

3. **Baselines (KDSC, TDBM)** — separate notebooks under `notebooks/KDSC/` and `notebooks/TDBM/`. Each has a `DATASET = 'av2' | 'waymo'` switch at the top. KDSC writes `style_label_kdsc`; TDBM writes `style_label_tdbm` (binary) and `tdbm_raw_style_label` (6-way). Outputs under `artifacts/kdsc_replication/{dataset}/` and `artifacts/tdbm_replication/{dataset}/`.

4. **GIF visualization** — `*_gifs*.ipynb` notebooks in the same folders sample agents per cluster and render the trajectory plus a dashboard (speed / relative-speed / long-accel / lat-accel).

5. **Comparison** — `notebooks/Comparison/`:
   - `per_method_stats.ipynb` — cluster sizes, prediction-error (`mtr_minfde6`) per cluster, feature histograms. Each method on its own row set, no inner join.
   - `clustering_comparison_continuous.ipynb` — continuous aggression scores per method, rank correlations, top-K% overlap, Cohen's d and KS statistic.
   - `clustering_comparison.ipynb` — binary join-based comparison (kept for completeness; misleading at unbalanced prevalence).

## Environment

GPU node example:

```bash
srun --pty --qos=huge-long --account=gamma --partition=gamma --time=01:59:00 \
     --ntasks=4 --mem=16gb --gres=gpu:rtxa5000:1 zsh
module load cuda/12.6.3 gcc/11.2.0
conda activate tailrisk-mp-cu126
```

If `validate_mtr_checkpoint.py` segfaults, the UniTraj CUDA extensions need rebuilding on the node:

```bash
bash scripts/rebuild_unitraj_mtr_ops.sh
```

## Datasets

- AV2 motion-forecasting (ScenarioNet-converted): `/fs/nexus-projects/pc_driving/datasets/argoverse2_sn`
- WOMD (ScenarioNet-converted): `/fs/nexus-projects/pc_driving/datasets/sn_womd`

## Report

LaTeX sources are in `report/` (one section per file). Figures referenced from the report are produced by the notebooks above and saved under each method's `artifacts/.../figures/` directory.
