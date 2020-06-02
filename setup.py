# Copyright 2020 Akamai Technologies, Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup, find_packages


with open("README.md", "r") as fp:
    long_description = fp.read()


setup(
    name="pallas",
    version="0.2",
    author="Miloslav Pojman",
    author_email="mpojman@akamai.com",
    description="AWS Athena client",
    keywords=["Amazon", "Athena", "AWS", "database", "db", "Jupyter", "Presto", "SQL"],
    license="Apache License 2.0",
    url="https://github.com/akamai/pallas",
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
        "Framework :: IPython",
        "Framework :: Jupyter",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: SQL",
        "Topic :: Database",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Typing :: Typed",
    ],
    zip_safe=False,
    include_package_data=True,
)
