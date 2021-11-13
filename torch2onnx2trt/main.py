import logging
import subprocess
from enum import Enum
from typing import Tuple

import onnx
import torch
from onnxsim import simplify
try:
    from pytorch_quantization import nn as quant_nn
    from pytorch_quantization import quant_modules
except ImportError as e:
    logging.warn('pytorch_quantization package is not installed. Conversion of int8 models is not available')


class PrecisionModes(Enum):
    fp32 = 'fp32', ''
    fp16 = 'fp16', '--fp16'
    int8 = 'int8', '--int8'
    # best = 'best', '--best'


def convert_torch2onnx(in_model: torch.nn.Module,
                       out_model_path: str,
                       input_shape: Tuple[int, ...],
                       input_names: Tuple[str] = ('input_0',),
                       output_names: Tuple[str] = ('output_0',),
                       opset_version: int = 13,
                       use_onnxsim: bool = True,
                       int8: bool = False
                       ):
    """
    This function converts PyTorch models to ONNX using both built-in pytorch functionality
    and onnxsim package. The usage of onnxsim is important if you want to port the
    model to TensorRT later.
    :param in_model: torch.nn.Module model that needs to be converted to ONNX
    :param out_model_path: Output path for saving resulting ONNX model
    :param input_shape: ONNX models expect input of fixed shape so you need to provide it
    :param input_names: Names for all of your model input nodes. You will need them later for
    TensorRT inference. By default the function expects model with one input and assigns
    the name 'input_0' to it
    :param output_names: Names for all of your model output nodes. You will need them later for
    TensorRT inference. By default the function expects model with one output and assigns
    the name 'output_0' to it
    :param opset_version: ONNX paramater that defines the set of operations used for parsing
    model to ONNX. By default opset is set to 9. I was able to get correct results for segmentation
    models using this older opset. Despite the ONNX warnings, opsets higher than 11 yielded
    models with incorrect result in contrast with opsets 9 and 10.
    :param use_onnxsim: Whether to use onnx-simplifier (https://github.com/daquexian/onnx-simplifier)
    package. Defaults to true because it is often needed for later TensorRT conversion.
    :param int8: Set to True, if the in_model parameter contains a model quantised via
    pytorch_quantization (https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization)
    """
    if int8:
        quant_nn.TensorQuantizer.use_fb_fake_quant = True
        quant_modules.initialize()

    device = torch.device('cpu')
    in_model.to(device)
    in_model.eval()

    input_ = torch.ones(input_shape)
    input_ = input_.to(device)

    print(f'Model input names: {input_names}')
    print(f'Model output names: {output_names}')
    print('Exporting model to ONNX...')
    torch.onnx.export(in_model, input_, out_model_path, export_params=True,
                      verbose=True, output_names=output_names,
                      input_names=input_names, opset_version=opset_version)

    # Simplify the ONNX network
    # Some network backbones, like ResNet, should work without it, but for EfficientNet you need it
    if use_onnxsim:
        model_onnx = onnx.load(out_model_path)
        model_simp, check = simplify(model_onnx)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, out_model_path)

    print(f'Model exported to: {out_model_path}')


def convert_onnx2trt(in_model: str,
                     out_model_path: str,
                     precision: str = 'fp16',
                     workspace: int = 2048,
                     ):
    """
    This function converts ONNX model to TensorRT using subprocess package and trtexec command line tool
    :param in_model: Path to ONNX model that needs to be converted to TensorRT
    :param out_model_path: Output path for saving resulting TensorRT model
    :param precision: What precision to use for model conversion.
    'fp32', 'fp16' and 'int8' are the available options.
    :param workspace: How much Mb of GPU memory to allocate for conversion
    """
    command = f'/usr/src/tensorrt/bin/trtexec --onnx={in_model} --explicitBatch --workspace={workspace} --saveEngine={out_model_path}'

    use_precision = None
    for mode in PrecisionModes:
        if mode.value[0] == precision:
            use_precision = mode.value[1]
            command = command + f' {use_precision}'
            break

    assert use_precision is not None, "Precision should be one of ['fp32', 'fp16']."
    subprocess.run(command, shell=True)
