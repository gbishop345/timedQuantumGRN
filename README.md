# timedQuantumGRN

This repository is a modified and extended version of the original [QuantumGRN](https://github.com/cailab-tamu/QuantumGRN) project. It is specifically tailored for **Time-Series Single-Cell RNA (scRNA-seq) data**, enabling the discovery of **directed, causal regulatory relationships** across sequential developmental stages.

## 🚀 Key Modifications & Enhancements

This fork introduces several critical changes to transition the original model from capturing undirected correlations to inferring **directed causal networks (DAGs)**.

### 1. Core Algorithmic Change: Directed Causal Weights
* **File Modified:** `qscgrn/optimizer.py`
* **Details:** In the original QuantumGRN, the gradient descent mechanism averaged the symmetric edges (`u -> v` and `v -> u`) to enforce an undirected graph. We removed this symmetric constraint in the `compute_gradient` function. The model now calculates independent directional weights, allowing $t_0$ genes to strictly drive $t_1$ genes without forced reciprocity.

### 2. Time-Asymmetric Training Initialization
* **Details:** We explicitly set `train_encoder=False` during the model training phase. This freezes the basal state (the starting time point, e.g., $0h$) and ensures the optimization strictly targets the transition probability to the next state ($1h$). This prevents the model from altering historical states to fit future data.

### 3. Matrix Dimension Fixes
* **Details:** Fixed a critical tensor dimension error (Maximum allowed dimension exceeded) by correcting the matrix shape handling during Pandas extraction. Removed `.T` transpositions to properly pass `(cells, genes)` shaped tensors into the quantum state simulator, allowing it to scale efficiently.

### 4. Advanced Directed Visualization
* **Details:** Replaced the default undirected visualizer with a custom `networkx`-based DAG visualizer. 
  * Explicit directional arrows (`-|>` ) are now rendered (resolving node overlap issues).
  * Green edges denote positive regulation; Red edges denote negative regulation.
  * Edge thickness linearly correlates with the absolute weight of the optimized interaction.

### 5. Automated Time-Course Pipeline
* **Files Added:** `test/run_timecourse_qgrn.py` & `test/run_causal_qgrn.py`
* **Details:** Built an automated pipeline to iterate through the Kouno (2013) macrophage differentiation dataset checkpoints: `0h -> 1h -> 6h -> 12h -> 24h -> 48h -> 72h -> 96h`. It automatically generates temporal network SVGs and weight CSVs into a `results_timecourse/` directory.

## 📁 Project Structure Additions

* `dataset/rna.csv`: The target time-series scRNA-seq dataset.
* `test/run_causal_qgrn.py`: Script for single-step transition analysis (e.g., 0h to 1h).
* `test/run_timecourse_qgrn.py`: Batch execution script for complete multi-stage time-series analysis.
* `quantumGRN_Install.sh`: Updated installation script pointing to this specific fork with corrected directory mappings.

## 💻 How to Run

1. Clone this repository and complete the installation via `quantumGRN_Install.sh`.
2. Ensure your data is located at `dataset/rna.csv` with a time column named `h`.
3. To run the full dynamic evolution pipeline, execute:
   ```bash
   cd test
   python run_timecourse_qgrn.py