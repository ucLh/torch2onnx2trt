from pathlib import Path
from setuptools import find_packages, setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='torch2onnx2trt',
    long_description=long_description,
    long_description_content_type='text/markdown',
    version='0.1.2',
    packages=find_packages(exclude='tests'),
    url='https://github.com/ucLh/torch2onnx2trt',
    license='MIT',
    author='Vladimir Luchinskiy',
    author_email='',
    description='A simple package that wraps PyTorch models conversion to ONNX and TensorRT',
    # dependency_links=['https://pypi.ngc.nvidia.com'],
    install_requires=[
        'onnx_simplifier>=0.3.6',
        'onnx>=1.9.0',
        # 'pytorch-quantization'
    ],
)
