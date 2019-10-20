from setuptools import setup, find_packages

setup(name='dqn',
      version='0.0.1',
      description='Multi-Agent Deep Deterministic Policy Gradient',
      url='https://github.com/Tilidaimon/deeprl/tree/master/multi-trainer/network',
      author='Igor Mordatch',
      author_email='mordatch@openai.com',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=['gym', 'numpy-stl']
)
