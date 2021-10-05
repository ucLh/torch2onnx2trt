from setuptools import find_packages, setup

setup(
    name='torch2onnx2trt',
    version='0.1',
    packages=find_packages(exclude='tests'),
    url='https://github.com/ucLh/torch2onnx2trt',
    license='MIT',
    author='Vladimir Luchinskiy',
    author_email='',
    description='A simple package that wraps PyTorch models conversion to ONNX and TensorRT',
    dependency_links=['https://pypi.ngc.nvidia.com'],
    install_requires=[
        'onnx_simplifier>=0.3.6',
        'onnx>=1.9.0',
        'pytorch-quantization>=2.1.0'
    ],
)
