from setuptools import setup

with open('README.md') as f:
    readme = f.read()

with open('LICENSE.md') as f:
    license = f.read()

try:
    from pypandoc import convert
    readme = convert(readme, 'rst', format='md')
    license = convert(license, 'rst', format='md')
except ImportError:
    print("Warning: pypandoc module not found, could not convert Markdown to RST")

setup(
    name='pykitti',
    version='0.1.0',
    description='A minimal set of tools for working with the KITTI dataset in Python',
    long_description=readme,
    author='Lee Clement',
    author_email='lee.clement@robotics.utias.utoronto.ca',
    url='https://github.com/utiasSTARS/pykitti',
    download_url='https://github.com/utiasSTARS/pykitti/tarball/0.1.0'
    license=license,
    packages=['pykitti'],
    install_requires=['numpy', 'matplotlib']
)
