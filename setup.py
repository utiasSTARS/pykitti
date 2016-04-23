from setuptools import setup

with open('README.md') as f:
    readme = f.read()

with open('LICENSE.md') as f:
    license = f.read()

setup(
    name='pykitti',
    version='0.0.1',
    description='A minimal set of tools for working with the KITTI dataset in Python',
    long_description=readme,
    author='Lee Clement',
    author_email='lee.clement@robotics.utias.utoronto.ca',
    url='https://github.com/utiasSTARS/pykitti',
    license=license,
    packages=['pykitti'],
    install_requires=['numpy', 'matplotlib']
)
