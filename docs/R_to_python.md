# From the original R code to this package

The original repository ([mandarsc/CRPAM](https://github.com/mandarsc/CRPAM)) has three R scripts, now moved to `R/`. They mix the published method with exploratory code and hard-coded Windows paths. This table shows what each part became.

## What the paper uses

| Paper (PMLR 92, 2018) | R source | Python |
|---|---|---|
| Random sparse/dense DAGs, `pcon = 2/(p-1)` (dense: ×2) | `randomDAG` (codeANM) via `genTrainData` / `genTestData` | `crpam.graphs.random_dag`, `default_edge_prob` |
| Linear SEM, coefs ±[0.1, 2], non-Gaussian noise | `randomB`, `sampleDataFromG(funcType="linear", noiseType="normalRandomVariancesRandomExp")` | `crpam.simulate.simulate_sem` with `SEMConfig.paper_linear()` |
| Nonlinear SEM, GP functions with bandwidth 1 | `sampleDataFromG(funcType="GAM", noiseType="normalRandomVariances")` | `simulate_sem` with `SEMConfig.paper_nonlinear()` |
| Residuals from OLS / GAM | `train_linear`, `train_gam` (mgcv) | `crpam.regression.LinearRegressor`, `AdditiveSplineRegressor` |
| Parent / mix-parent / non-parent regressor sets (r_non=3, s_non≤4, r_mix=2, s_mix≤3) | `genSynData.R::genTrainData` | `crpam.training.regressor_sets_for_child`, `generate_training_data` |
| Kernel mean embedding features, γ = 1, 3n features | `kernelMatrix(rbfdot(sigma=1), ·)` + `apply(·, 2, mean)` | `crpam.features.KernelMeanEmbedding` |
| Balanced training set, RF with 501 trees | `genSynData.R::trainClassifier` | `crpam.training.balanced_training_set`, `CRPAM.fit_simulated` |
| Candidate sets of size 1..s_test (all if p<10, 5 if p=10, 3 if p>10) | `genTestData` | `crpam.inference.candidate_sets`, `default_max_candidate_size` |
| Frequency heuristic (>50% of predicted parent sets) | `method2.R::buildEstAmat`, heuristic 3 (CRP-CAM-3) | `crpam.inference.frequency_heuristic` (`heuristic="frequency"`) |
| SHD, SID, d, FPR | `method2.R::getPerformanceMetrics` (pcalg, SID) | `crpam.metrics` |
| Baselines RESIT, LiNGAM, PC, GES, Random | `runCDMethods` | `crpam.baselines` |
| Feature mean/sd plot | `plotKernelstats` | `experiments/plot_feature_stats.py` |
| Experiment loop and result tables | `method2.R` top-level loop, `predictCR` | `experiments/run_synthetic.py` |

## Exploratory code, kept as options

| R source | What it was | Python |
|---|---|---|
| `genSynDataApprox.R::genWeights`, `genKernelFeatures` | Random Fourier feature approximation, γ ∈ {0.15, 1.5, 15}, 100 features each | `crpam.features.RandomFourierEmbedding` |
| `genSynDataApprox.R::genTrainData` / `genTestData` | Median-heuristic bandwidths | `KernelMeanEmbedding(bandwidth="median")` |
| `method2.R::buildEstAmat` heuristic 1 (CRP-CAM-1) | Pick one predicted parent set | `heuristic="max_probability"` (cleaned up, see below) |
| `method2.R::performValidation`, `genSynData.R::performValidation`/`performSampling` | Tune the kernel and class ratio on one validation graph | Not ported as code. Use the `mix_ratio` / `non_parent_ratio` arguments of `CRPAM` and loop over them |

## Dropped

* `method2.R::getKernelMatrix`, `getNoPaKernelMatrix`, `getPredKernelMatrix`, `getHSIC`: early file-based versions of the feature code that read intermediate `.txt` files. The in-memory pipeline replaces them.
* `genSynDataApprox.R::genANMData`: plain data dump. Use `simulate_sem`.
* Stray top-level snippets (plots, `cd_results <- ...`) that referenced variables defined elsewhere.
* GDS and MMHC baselines: no maintained Python implementation.

## Bugs and quirks found in the R code

1. **`buildEstAmat`, heuristic 1:** `pa` is computed from the max-score sets and then overwritten by `unlist(all_pa[min(which(var_pa_score==1))])`. That indexes the list of *predicted* sets with an index into *all* sets, so it returns the wrong set. `max_probability` instead takes the highest-probability set, with ties going to the smallest set.
2. **Heuristic 2** was commented out, so CRP-CAM-2 always returned an empty graph.
3. **Mix-parent sets** were generated only inside `if (length(no_pa_c) > 0)`. A child with no non-parents never got mix sets. The Python version builds the two kinds independently.
4. **`trainClassifier` sampling:** `sample(1:nrow(mix), nrow(pa))` fails if there are fewer mix rows than parent rows. The Python version samples with replacement in that case.
5. **`genSynDataApprox.R::genTrainData`** computes bandwidths from `gp_fit$residuals` before `gp_fit` exists, and calls `scale(X[[n]])` outside the loop. That file was never run end to end, which is why it is only kept as options.
6. **`method2.R`** calls `buildEstAmat(test_pred_prob, nrow(test_data), p)` without the `max_pa_test`/`thresh` arguments.
7. **Sample-order dependence:** the paper's features are the kernel mean embedding *evaluated at the n data points*, so feature k means "embedding at sample k". The value depends on row order, and train and test must have the same n. `KernelMeanEmbedding(sort=True)` and `RandomFourierEmbedding` remove this dependence and are worth testing.
8. **The heuristics can return cycles.** `enforce_dag=True` removes the weakest edge on each cycle (new; not in the paper).

## Known differences from the R implementation

* `mgcv::gam` is replaced by additive cubic B-splines with a GCV-chosen ridge penalty. Residuals are close but not identical.
* The data generators were checked line by line against codeANM ([bquast/ANM](https://github.com/bquast/ANM/tree/master/codeANM), the code the paper cites). Two details matter:
  * In `sampleDataFromGAMGP`, nodes with parents get noise `0.2 * sqrt(v) * N(0,1)`, while source nodes get `sqrt(v) * N(0,1)`, with `v ~ U[1, 2]`. The first version of this port left out the 0.2. That made the nonlinear data far noisier than the paper's, and CRPAM's SHD rose from about 4 to about 9.
  * In `sampleDataFromGLinear` the noise is `E * sign(z') * |z|^a` with `E ~ U[0.1, 0.5]` used directly as a scale (not a variance), and `a ~ U[2, 4]`.
* The GAM basis size follows `train_gam`: k = 10 per smooth term, reduced to `ceil(n / (3d))` when `n/d < 30`.
* `pcalg::shd` and `SID::structIntervDist` were re-implemented. Unit tests cover hand-checked cases. SID is reported only when the estimate is a DAG (it is `NaN` for CPDAGs and cyclic graphs). The R package reports lower and upper bounds for CPDAGs.
