from setuptools import setup, find_packages

setup(
    name="learnspec",
    version="0.5.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "tensorflow>=2.8.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "umap-learn>=0.5.0",
        "hdbscan>=0.8.0",
        "specutils>=1.8.0",
        "astropy>=5.0.0",
    ],
    python_requires=">=3.9",
    author="Aayush Saxena",
    author_email="aayush.saxena@physics.ox.ac.uk",
    description="A package for analyzing astronomical spectra using VAEs",
    keywords="astronomy, spectra, machine learning, VAE",
)