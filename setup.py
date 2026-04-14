from setuptools import setup, find_packages

setup(
    name="danooc",
    version="0.1.0",
    description="CNN image classifier trained on CIFAR-10",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "Pillow>=9.0.0",
    ],
)
