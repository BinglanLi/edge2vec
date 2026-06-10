# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

edge2vec is a two-stage pipeline for learning node embeddings in heterogeneous (multi-edge-type) networks, described in [Gao et al. 2019, BMC Bioinformatics](https://doi.org/10.1186/s12859-019-2914-x).

## Environment setup

```bash
conda env create -f environment.yml
conda activate edge2vec
```

## Running the pipeline

**Step 1 — compute edge transition matrix** (`transition.py`):
```bash
python transition.py --input data.csv --output matrix.txt --type_size 3 --em_iteration 5 --e_step 3 --walk-length 3 --num-walks 2
```

**Step 2 — generate node embeddings** (`edge2vec.py`):
```bash
python edge2vec.py --input data.csv --matrix matrix.txt --output vector.txt --dimensions 128 --walk-length 3 --num-walks 2 --p 1 --q 1
```

Sample inputs: `data.csv`, `unweighted_graph.txt`, `weighted_graph.txt`. Sample outputs: `matrix.txt`, `vector.txt`.

## Architecture

The pipeline is split across two scripts that must be run in sequence:

**`transition.py`** — learns an `N×N` edge-type transition matrix (where `N = --type_size`) using EM:
- **M step**: `simulate_walks` generates edge-level random walks starting from each edge, biased by the current transition matrix and the node2vec `p`/`q` hyperparameters.
- **E step**: `update_trans_matrix` compares per-type edge-frequency vectors across walks using one of four similarity metrics (controlled by `--e_step`): 1=Wilcoxon, 2=entropy, 3=Spearmanr, 4=Pearsonr. Note: metrics 2–4 currently fall back to Wilcoxon internally.
- Output: a space-delimited `matrix.txt`.

**`edge2vec.py`** — runs node-level biased random walks guided by the transition matrix, then trains a Word2Vec skip-gram model (gensim) on the walk sequences to produce node embeddings:
- `simulate_walks` / `edge2vec_walk`: node-level walks; transition probability between consecutive edges is `matrix[prev_edge_type][next_edge_type] × edge_weight`, scaled by `p` (return) or `q` (explore).
- Embeddings saved in word2vec text format via `model.wv.save_word2vec_format`.

**Input format**: space-separated edge list. Unweighted: `src dst type id`. Weighted: `src dst type weight id`. Edge types are 1-indexed integers; the matrix uses 0-indexed access (`type - 1`).

**`util.py`**: miscellaneous helper utilities (not imported by the two main scripts by default).
