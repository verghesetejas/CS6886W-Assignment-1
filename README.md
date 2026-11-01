# CS6886W Assignment 1 — VGG6 / CIFAR-10

Modular PyTorch starter project for the assignment. It uses the professor's VGG6 channel configuration `[64, 64, M, 128, 128, M]`, BatchNorm, adaptive average pooling and a 10-class classifier.

## Setup (Windows PowerShell)
Use Python 3.12 in a virtual environment. Install a CUDA-enabled PyTorch build from https://pytorch.org/get-started/locally/ and then:
```powershell
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Run
```powershell
python -m training.train --config configs/baseline.yaml
python -m training.evaluate --checkpoint outputs/best.pt
```
W&B:
```powershell
wandb login
python -m experiments.run_sweep --create --count 30
```

CIFAR-10 downloads into `data/`. The official training set is split reproducibly into 45,000 train and 5,000 validation examples. The official test set is held out for final evaluation. Training augmentation: random crop with 4-pixel padding, random horizontal flip, tensor conversion, channel normalization, and one 16×16 Cutout mask. Evaluation uses tensor conversion and normalization only.

Default seed is 42. Results are not pre-filled: run the experiments and report measured values. Keep the best validation checkpoint and evaluate the selected model on the test set only after configuration selection. Upload the source and final selected checkpoint to GitHub; do not upload dataset files.
