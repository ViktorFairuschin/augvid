from distutils.core import setup
from vizooal import __version__


setup(
    name='vizooal',
    version=__version__,
    packages=['vizooal'],
    url='https://github.com/ViktorFairuschin/vizooal.git',
    license='MIT Licence',
    author='Viktor Fairuschin',
    author_email='',
    description='A collection of video augmentation tools based on Keras.'
)
