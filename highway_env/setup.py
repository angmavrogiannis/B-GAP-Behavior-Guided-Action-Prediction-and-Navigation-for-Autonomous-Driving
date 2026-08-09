from setuptools import setup, find_packages

setup(
    name="highway-env",
    version="1.4.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "gym",
        "pygame",
    ],
    python_requires=">=3.6",
)
