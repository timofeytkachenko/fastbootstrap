from setuptools import find_packages, setup


def readme():
    with open("README.md", "r") as f:
        return f.read()


setup(
    name="fastbootstrap",
    version="1.2.4",
    author="Timofey Tkachenko",
    author_email="timofey_tkachenko@pm.me",
    description="Fast Python implementation of statistical bootstrap",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/timofeytkachenko/fastbootstrap",
    packages=find_packages(),
    install_requires=[
        "jupyter==1.0.0",
        "ipython==8.21.0",
        "matplotlib==3.8.3",
        "plotly==5.19.0",
        "numpy==1.26.4",
        "scipy==1.12.0",
        "pandas==2.2.0",
        "tqdm==4.67.1",
        "seaborn==0.13.2",
        "joblib==1.4.2",
        "rich==13.9.4",
        "markdown-it-py==2.2.0",
    ],
    keywords="bootstrap fast_bootstrap",
    python_requires=">=3.10",
)
