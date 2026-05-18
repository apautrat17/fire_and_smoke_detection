# Fire and Smoke Detection

The goal of the project is to train a model to detect whether an image contains fire, smoke, or both.
The model should then be used in real time with a camera and possibly embedded in a robot.

The model is trained with the [DFireDataset](https://github.com/gaia-solutions-on-demand/DFireDataset).

> tensorboard --logdir ./runs/training

## Example

## Installation

```bash
python -m venv venv
.\venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

```bash
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
```

## Test