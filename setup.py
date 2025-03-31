from setuptools import setup, find_packages
from augvid import __version__

setup(
    name='augvid',
    version=__version__,
    description='A collection of video augmentation layers',
    # long_description=long_description,
    # long_description_content_type='text/markdown',
    url='https://github.com/ViktorFairuschin/augvid',
    author='Viktor Fairuschin',
    # author_email='author@example.com',
    packages=find_packages(where='augvid'),
    python_requires='>=3.7, <4',
    install_requires=[
        'tensorflow',
        'keras',
        'numpy',
    ],
    extras_require={
        'dev': [
            'decord',
            'opencv-python',
        ],
    },
)