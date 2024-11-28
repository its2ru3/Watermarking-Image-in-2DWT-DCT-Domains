# setup.py

from setuptools import setup, find_packages

setup(
    name="wm2dwt",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        # add dependencies here
        # eg 'numpy>=1.11.1'
    ],
    entry_points={
        'console_scripts': [
            'wm = wm2dwt.main:wm2dwt',  # This defines the terminal command
        ],  
    },
)