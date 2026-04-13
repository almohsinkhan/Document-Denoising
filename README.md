# Document Denoising

This project trains convolutional encoder-decoder models to remove noise from grayscale document images. The current workflow lives in the notebook `pipline.ipynb`, where the dataset is loaded, multiple U-Net variants are trained, and denoised outputs are visualized.

## Overview

- Goal: restore noisy document scans to cleaner grayscale versions
- Input data: paired noisy and clean images stored in `data/train` and `data/train_cleaned`
- Dataset size in this repo: 144 noisy images and 144 matching clean images
- Image preprocessing: grayscale conversion, resize to `128 x 128`, tensor conversion
- Validation split: `80/20` with a fixed seed
- Saved checkpoint: `best_model_v2.pth`

The notebook experiments with:

- A baseline U-Net trained with `MSELoss`
- An improved U-Net using a combined `MSE + L1` loss
- Attention U-Net variants for stronger skip-connection feature selection

## Project Structure

```text
Document-Denoising/
├── best_model_v2.pth
├── pipline.ipynb
├── requirements.txt
├── README.md
└── data/
    ├── sampleSubmission.csv
    ├── train/
    └── train_cleaned/
```

## Requirements

- Python 3.10+
- `pip`
- Optional but recommended: a virtual environment
- Optional: CUDA-enabled GPU for faster training

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

To work in the notebook:

```bash
jupyter notebook
```

Then open `pipline.ipynb`.

## Workflow In The Notebook

The notebook currently follows this sequence:

1. Import dependencies and detect the available device (`cuda` or `cpu`).
2. Build a custom `DocumentDataset` that pairs files from `data/train` and `data/train_cleaned`.
3. Apply transforms:
   - convert to grayscale
   - resize to `128 x 128`
   - convert to tensor
4. Split the dataset into training and validation sets.
5. Train several model variants for `10` epochs.
6. Save the best-performing model weights to `best_model_v2.pth`.
7. Visualize noisy inputs, predictions, binary outputs, and ground-truth images.

## Running Training

Training is notebook-based right now, so the simplest way to run the project is:

1. Install dependencies.
2. Launch Jupyter.
3. Open `pipline.ipynb`.
4. Run the cells in order.

## Notes

- The notebook filename is currently `pipline.ipynb` in the repository.
- The code uses grayscale images only.
- The best checkpoint filename is reused across notebook experiments, so retraining may overwrite `best_model_v2.pth`.
- There is no standalone training or inference script yet; the notebook is the source of truth for the workflow.

## Future Improvements

- Move model definitions and training logic into Python modules
- Add a standalone inference script for denoising new images
- Track metrics such as PSNR or SSIM during validation
- Add argument-based training scripts for reproducibility
