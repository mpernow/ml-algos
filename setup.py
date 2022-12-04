from setuptools import setup, find_packages
from os import path

__version__ = '0.0.1'

root_dir = path.abspath(path.dirname(__file__))

# get the dependencies and installs
with open(path.join(root_dir, 'requirements.txt'), 'r') as f:
    requirements = f.read().split('\n')

setup(
    name='mlalgos',
    version=__version__,
    description='Implementation of some machine learning algorithms.',
    url='https://github.com/mpernow/ml-algos',
    download_url='https://github.com/mpernow/ml-algos/tarball/master',
    license='MIT',
    author='Marcus Pernow',
    author_email='marcus.pernow@gmail.com',
    packages=find_packages(),
    install_requires=requirements
)