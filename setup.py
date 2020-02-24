from setuptools import setup, find_packages


with open("README.md", "r") as fp:
    long_description = fp.read()


setup(
    name="pallas",
    version="0.0.dev",
    author="Miloslav Pojman",
    author_email="mpojman@akamai.com",
    description="AWS Athena client",
    keywords=["Amazon", "Athena", "AWS", "database", "db", "Jupyter", "Presto", "SQL"],
    license="...",  # TODO
    url="...",  # TODO
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages("src"),
    package_dir={"": "src"},
    python_requires=">=3.7",
    install_requires=["boto3"],
    extras_require={
        "dev": ["flake8", "pytest", "mypy"],
        "pandas": ["pandas>=1.0.0"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Framework :: IPython",
        "Framework :: Jupyter",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: SQL",
        "Topic :: Database",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Typing :: Typed",
    ],
    zip_safe=False,
)
