import unittest

import numpy as np
import onnxruntime as rt
import torch
from segmentation_models_pytorch import Unet

from torch2trt import convert_torch2onnx


class TestConversion(unittest.TestCase):
    def testOnnxConversion(self):
        # Prepare dummy input
        input_size = (1, 3, 640, 1280)
        dummy_input = np.ones(input_size).astype(np.float32)

        # Convert to onnx
        onnx_model_path = 'effnetb0_unet_gray_2grass_iou55.onnx'
        model = Unet(encoder_name='efficientnet-b0', classes=11)
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
        self.assertLess(np.mean(diff), 0.01)
