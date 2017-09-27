from setuptools import setup

setup(name='lsm',
      version='0.1',
      description='Liquid state machines (LSM) for spiking neural networks (SNN).',
      #  url='...',  # TODO
      author='Michael Hoff',
      author_email='mail@michael-hoff.net',
      license='MIT',  # TODO
      packages=['lsm'],
      install_requires=[  # TODO check dependencies
        'numpy',
        'scipy',
        'nest',  # TODO use `dependency_links` instead
      ],
      zip_safe=False,
      )
