from setuptools import setup, find_packages


def readme():
    with open('README.md', 'r') as f:
        return f.read()


setup(
    name='fastbootstrap',
    version='1.0.0',
    author='Timofey Tkachenko',
    author_email='timofey_tkachenko@pm.me',
    description='Fast Python implementation of statistical bootstrap',
    long_description=readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/timofeytkachenko/fastbootstrap',
    packages=find_packages(),
    install_requires=['jupyter==1.0.0'
                      'ipython==8.12.0',
                      'matplotlib==3.7.1',
                      'plotly==5.14.1',
                      'seaborn==0.12.2',
                      'numpy==1.24.3',
                      'scipy==1.10.1',
                      'pandas==1.5.3',
                      'tqdm==4.65.0',
                      'multiprocess==0.70.14'],
    keywords='bootstrap resampling fast_bootstrap, fast_resampling',
    python_requires='>=3.10.12'
)
