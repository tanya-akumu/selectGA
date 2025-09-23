from setuptools import setup, find_packages

setup(
    name="selectga_estimation",
    version="0.1.0",
    description="A package for GA estimation from blindsweep videos",
    author="Tanya Akumu",
    packages=find_packages(),
    install_requires=[
        # Add your dependencies here, e.g. 'numpy', 'pandas'
    ],
    python_requires='>=3.6',
)