from os import path

import setuptools

path_to_repo = path.abspath(path.dirname(__file__))
with open(path.join(path_to_repo, 'readme.md'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name="dvu",
    version="0.0.2",
    author="Chandan Singh",
    author_email="chandan_singh@berkeley.edu",
    description="Functions for data visualization in matplotlib.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/csinva/data-viz-utils",
    packages=setuptools.find_packages(exclude=['tests']),
    install_requires=[
        'numpy',
        'matplotlib',
        'seaborn',
        'scikit-learn>=0.23.0',  # 0.23+ only works on py3.6+
        'adjustText',
    ],
    python_requires='>=3.6',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
