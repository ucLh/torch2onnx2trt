### Convert PyTorch models to ONNX and then to TensorRT
[![Build Status](https://app.travis-ci.com/ucLh/torch2trt.svg?branch=master)](https://app.travis-ci.com/ucLh/torch2trt)

#### Requirements
1) PyTorch (tested with 1.5)
2) TensorRT (tested with 8.0)
3) The rest of the requirements are listed in `requirements.txt` and can
 be installed automatically via `pip` installation
 
#### Installation
There is `setup.py` file in the repo, so the installation is pretty 
straightforward
```
git clone https://github.com/ucLh/torch2trt.git
cd torch2trt
pip3 install -e ./
```

#### Usage
```python
from torch2trt import convert_torch2onnx, convert_onnx2torch
# Load your model
model = ...
# You need to pass your model with loaded weights, an output path for onnx model
# and desired input shape to convert_torch2onnx function
convert_torch2onnx(model, 'effnetb0_unet_gray_2grass_iou55.onnx', (1, 3, 640, 1280))
# convert_onnx2torch expects a path to onnx model and an output path for resulting
# TensorRT .bin model
convert_onnx2torch('../effnetb0_unet_gray_2grass_iou55.onnx', '../effnetb0_unet_gray_2grass_iou55.bin')
```
