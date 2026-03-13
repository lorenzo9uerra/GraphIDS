[![arXiv](https://img.shields.io/badge/arXiv-Preprint-b31b1b.svg)](https://arxiv.org/abs/2509.16625) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# Self-Supervised Learning of Graph Representations for Network Intrusion Detection

This repository provides the official code for our paper, accepted at NeurIPS 2025.

## Overview

GraphIDS is a self-supervised intrusion detection system that learns graph representations of normal network traffic patterns. The model combines:
- **E-GraphSAGE**: An inductive GNN that embeds each flow with its local topological context
- **Transformer Autoencoder with Attention Masking**: Reconstructs flow embeddings while learning global co-occurrence patterns

Flows with high reconstruction errors are flagged as potential intrusions. By jointly training both components end-to-end, the model achieves state-of-the-art performance on NetFlow benchmarks (up to 99.98% PR-AUC and 99.61% macro F1-score).

<p align="center">
  <img src="figures/full_pipeline.png" alt="Graph representation learning process">
</p>

*Note: This implementation uses PyTorch Geometric (PyG) for improved maintainability and broader compatibility. For exact reproduction of the paper results and pretrained models, see the [DGL branch](https://github.com/lorenzo9uerra/GraphIDS/tree/dgl).*


## Requirements

### Installation

This project uses [uv](https://docs.astral.sh/uv/) for project management.

Sync the project dependencies:

```bash
uv sync
```

This will create a virtual environment and install all dependencies specified in `pyproject.toml`.

**Note:** The project is configured for CUDA 12.8. If you need a different CUDA version, modify the `[tool.uv.index]` URL in `pyproject.toml`.

### Datasets

The datasets can be downloaded from this website: https://staff.itee.uq.edu.au/marius/NIDS_datasets/

After downloading each dataset zip file, unzip it with the following command:

```bash
unzip -d <dataset_name> -j <filename>.zip
```

For example, for the NF-UNSW-NB15-v3 dataset:
```bash
unzip -d NF-UNSW-NB15-v3 -j f7546561558c07c5_NFV3DATA-A11964_A11964.zip
```

NOTE: The authors recently renamed the file for the NF-CSE-CIC-IDS2018-v2 and NF-CSE-CIC-IDS2018-v3 datasets as NF-CICIDS2018-v2 and NF-CICIDS2018-v3.\
To keep a consistent naming convention with the literature, the code expects the dataset directory and the dataset CSV file to be named as one of the 4 considered datasets: `NF-UNSW-NB15-v2`, `NF-UNSW-NB15-v3`, `NF-CSE-CIC-IDS2018-v2`, `NF-CSE-CIC-IDS2018-v3`.

## Quick Start

Once you have installed the dependencies and downloaded a dataset, you can train GraphIDS with:

```bash
uv run main.py --data_dir data/ --config configs/NF-UNSW-NB15-v3.yaml
```

This will train the model on the NF-UNSW-NB15-v3 dataset and automatically evaluate it after training.

## Training

### Experiment Tracking

We use Weights & Biases for experiment tracking. W&B is set to offline mode by default—no login is required, and all logs are stored locally. To enable online mode, pass the `--wandb` flag.

### Running Training

To train GraphIDS, run this command:

```bash
uv run main.py --data_dir <data_dir> --config configs/<dataset_name>.yaml
```
`<data_dir>` should point to the directory containing all the datasets. The code expects the directory structure found in the zip files (i.e., each CSV file should be located at `<data_dir>/<dataset_name>/<dataset_name>.csv`). For example, for the following directory structure:
```
data/
└── NF-UNSW-NB15-v3
    ├── FurtherInformation.txt
    ├── NF-UNSW-NB15-v3.csv
    ├── NetFlow_v3_Features.csv
    ├── bag-info.txt
    ├── bagit.txt
    ├── manifest-sha1.txt
    └── tagmanifest-sha1.txt
configs/
└── NF-UNSW-NB15-v3.yaml
```
You should run:
```bash
uv run main.py --data_dir data/ --config configs/NF-UNSW-NB15-v3.yaml
```

To specify different training parameters, you can either modify the configuration file in the `configs/` directory, or provide all parameters using command-line arguments. The full list of possible arguments can be accessed by running the command:
```bash
uv run main.py --help
```

## Evaluation

By running the command above, the model would also be evaluated after training. However, to only evaluate the model from a saved checkpoint, run the following command:

```bash
uv run main.py --data_dir <data_dir> --config configs/<dataset_name>.yaml --checkpoint checkpoints/GraphIDS_<dataset_name>_<seed>.ckpt --test
```

## Development

If you plan to modify the code, we recommend installing the development dependencies:

```bash
uv sync --extra dev
```

This installs `ruff` for linting and formatting, `ty` for type checking, and
`pre-commit` hooks.

To set up the git hooks:

```bash
uv run pre-commit install
```

To run the linter and formatter manually:

```bash
uv run ruff check .
uv run ruff format .
uv run ty check .
```

### Testing

Run the suite with:

```bash
uv run pytest
```

Pytest is configured in `pyproject.toml`. The suite uses synthetic data and is
intended to validate the preprocessing, training, and evaluation pipeline
without requiring the full datasets.

## Results

Our model achieves the following performance on the following datasets:

### [NF-UNSW-NB15-v3](https://rdm.uq.edu.au/files/abd2f5d8-e268-4ff0-84fb-f2f7b3ca3e8f)

| Model name         |  Macro F1-score  |  Macro PR-AUC  |
| ------------------ | ---------------- | -------------- |
| GraphIDS           |      99.61%      |      99.98%    |

### [NF-CSE-CIC-IDS2018-v3](https://rdm.uq.edu.au/files/4ac221b1-6bd6-42b1-bdf7-03f4fc7efb22)

| Model name         |  Macro F1-score  |  Macro PR-AUC  |
| ------------------ | ---------------- | -------------- |
| GraphIDS           |      94.47%      |      88.19%    |

### [NF-UNSW-NB15-v2](https://rdm.uq.edu.au/files/8c6e2a00-ef9c-11ed-827d-e762de186848)

| Model name         |  Macro F1-score  |  Macro PR-AUC  |
| ------------------ | ---------------- | -------------- |
| GraphIDS           |      92.64%      |      81.16%    |

### [NF-CSE-CIC-IDS2018-v2](https://rdm.uq.edu.au/files/ce5161d0-ef9c-11ed-827d-e762de186848)

| Model name         |  Macro F1-score  |  Macro PR-AUC  |
| ------------------ | ---------------- | -------------- |
| GraphIDS           |      94.31%      |      92.01%    |

The results are averaged over multiple seeds.

## Citation

If you find this work useful in your research, please consider citing our paper:

```bibtex
@inproceedings{guerra2025selfsupervised,
  title={Self-Supervised Learning of Graph Representations for Network Intrusion Detection},
  author={Lorenzo Guerra and Thomas Chapuis and Guillaume Duc and Pavlo Mozharovskyi and Van-Tam Nguyen},
  booktitle={Advances in Neural Information Processing Systems},
  volume={38},
  year={2025},
  url={https://openreview.net/forum?id=5bu1IOOvf0}
}
```

## License

All original components of this repository are licensed under the [Apache License 2.0](./LICENSE). Third-party components are used in compliance with their respective licenses.
