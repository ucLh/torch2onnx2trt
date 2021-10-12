import unittest

import numpy as np
import onnxruntime as rt
import torch
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules
from pytorch_quantization.tensor_quant import QuantDescriptor
from segmentation_models_pytorch import Unet

from torch2onnx2trt import collect_stats, compute_amax, convert_torch2onnx


class TestConversion(unittest.TestCase):
    def testOnnxConversion(self):
        # Prepare dummy input
        input_size = (1, 3, 640, 1280)
        dummy_input = np.ones(input_size).astype(np.float32)

        # Convert to onnx
        onnx_model_path = 'dummy.onnx'
        model = Unet(encoder_name='efficientnet-b0', classes=1)
        model.encoder.set_swish(memory_efficient=False)
        convert_torch2onnx(model, onnx_model_path, input_size)

        # Infer onnx model
        sess = rt.InferenceSession(onnx_model_path)
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
        onnx_pred = sess.run([output_name], {input_name: dummy_input})[0]

        # Infer pytorch model
        torch_pred = model(torch.from_numpy(dummy_input)).detach().cpu().numpy()
        diff = np.abs(onnx_pred - torch_pred)

        # Compare results. They are not very close
        self.assertLess(np.mean(diff), 0.02)

    def testOnnxInt8Conversion(self):
        # Use histogram calibrator as default one
        quant_desc_input = QuantDescriptor(calib_method='histogram')
        quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
        quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)

        # Initialize quantized modules
        quant_modules.initialize()

        # Create fake calibration data
        input_size = (1, 3, 640, 1280)
        dummy_input = torch.ones(input_size, dtype=torch.float32)

        # Create model
        model = Unet(encoder_name='resnet18', classes=1)

        # Calibrate model
        with torch.no_grad():
            collect_stats(model, dummy_input)
            compute_amax(model, method="percentile", percentile=99.99)

        # Convert to onnx
        onnx_model_path = 'dummy_int8.onnx'
        convert_torch2onnx(model, onnx_model_path, input_size, int8=True)

        # Infer onnx model
        sess = rt.InferenceSession(onnx_model_path)
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
        onnx_pred = sess.run([output_name], {input_name: dummy_input.numpy()})[0]

        # Infer pytorch model
        torch_pred = model(dummy_input).detach().cpu().numpy()
        diff = np.abs(onnx_pred - torch_pred)

        # Compare results. They are not very close
        self.assertLess(np.mean(diff), 0.02)
