from setuptools import setup, find_packages

setup(
    name='Aanalysis', # This name abbreviate antibody analysis
    version='0.0.1',
    description='desa paper python package',
    author='Danial Senejohnny',
    author_email='d.mohammadisenejohnny@umcutrecht.nl',
    packages=find_packages(),
    tests_require=['pytest'],
    python_requires='>=3.8',
)