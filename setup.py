from setuptools import setup, find_packages

with open('README.md') as file:
    long_description = file.read()
setup(
    name='LAD',
    version='0.1.0',
    py_modules=['LAD'],
    description='Python pacakge for extrapolating lake-area distribution to non-inventoried lakes and calculating vegetation coverage and methane emissions.',
    long_description=long_description,
    packages=find_packages(),
    python_requires='>=3.3',
    install_requires=[
            'geopandas',
            'matplotlib',
            'matplotlib-label-lines',
            'numpy',
            'pandas',
            'pyogrio',
            'scipy',
            'seaborn',
            'openpyxl',
            'statsmodels',
            'scikit-learn'
    ],
    url='https://github.com/ekcomputer/lake-area',
    license='MIT',
    author='Ethan Kyzivat',
    author_email='ethan_kyzivat@alumni.brown.edu',
)
