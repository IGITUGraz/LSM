from setuptools import setup, find_packages

setup(name='lsm',
      version='0.1',
      description='Liquid state machines (LSM) for spiking neural networks (SNN).',
      url='https://github.com/IGITUGraz/LSM',
      author='Michael Hoff',
      author_email='mail@michael-hoff.net',
      packages=find_packages(),
      license='GPLv3',
      install_requires=[
        'numpy',
        'scipy'
      ],
      zip_safe=False,
      )
