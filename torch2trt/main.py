import subprocess
from enum import Enum

import onnx
import torch
from onnxsim import simplify


class PrecisionModes(Enum):
    fp32 = 'fp32', ''
    fp16 = 'fp16', '--fp16'
    # int8 = 'int8', '--int8'


def convert_torch2onnx(in_model, out_model_path, input_shape, input_names=('input_0',),
                       output_names=('output_0',), opset_version=9):
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

    # Simplify the onnx network, it is needed for later TensorRT conversion
    # Some network backbones, like ResNet, should work without it, but for EfficientNet you need it
    model_onnx = onnx.load(out_model_path)
    model_simp, check = simplify(model_onnx)
    assert check, "Simplified ONNX model could not be validated"

    onnx.save(model_simp, out_model_path)

    print(f'Model exported to: {out_model_path}')


def convert_onnx2torch(in_model, out_model_path, precision='fp16', workspace=2048):
    command = f'/usr/src/tensorrt/bin/trtexec --onnx={in_model} --explicitBatch --workspace={workspace} --saveEngine={out_model_path}'

    use_precision = None
    for mode in PrecisionModes:
        if mode.value[0] == precision:
            use_precision = mode.value[1]
            command = command + f' {use_precision}'
            break

    assert use_precision is not None, "Precision should be one of ['fp32', 'fp16']."
    subprocess.run(command, shell=True)
