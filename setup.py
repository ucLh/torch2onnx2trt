from setuptools import setup

setup(
    name='torch2trt',
    version='0.1',
    packages=['torch2trt'],
    url='',
    license='MIT',
    author='Vladimir Luchinskiy',
    author_email='',
    description='A simple package that wraps PyTorch models conversion to ONNX and TensorRT',
    install_requires=[
        'onnx_simplifier>=0.3.6',
        'onnx>=1.9.0',
    ],
)
