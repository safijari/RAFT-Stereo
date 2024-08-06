from setuptools import setup, find_packages

setup(
    name="raftstereo",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch>=1.13.0",
        "torchvision>=0.15.1",
        "matplotlib",
        "tensorboard",
        "scipy",
        "opencv-python",
        "tqdm",
        "opt_einsum",
        "imageio",
        "scikit-image",
        "py7zr",  # p7zip equivalent for Python
    ],
    extras_require={
        "cuda": ["cudatoolkit==10.2.89"],
    },
)