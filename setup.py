from setuptools import setup, find_packages

setup(
    name="physics-pinn",
    version="0.1.0",
    description="Physics-Informed Neural Networks for classical mechanics",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "app": [
            "streamlit>=1.30.0",
            "plotly>=5.18.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "jupyter>=1.0.0",
        ],
    },
)
