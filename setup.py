
from setuptools import setup, find_packages


with open("README.rst", "r") as fp:
    long_description = fp.read()


setup(
    name="pallas",
    version="0.0.dev",
    author="Miloslav Pojman",
    author_email="mpojman@akamai.com",
    description="Convenient Facade to AWS Athena",
    long_description=long_description,
    packages=find_packages("src"),
    package_dir={"": "src"},
    python_requires=">=3.7",
    install_requires=[
        "boto3",
    ],
    extras_require={
        "dev": [
            "flake8",
            "pytest",
            "mypy",
        ],
        "pandas": [
            "pandas>=1.0.0",
        ],
    }
)
