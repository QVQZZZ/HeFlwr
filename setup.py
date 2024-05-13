from setuptools import setup, find_packages


def readme():
    with open("README.rst", encoding="UTF-8") as f:
        return f.read()


setup(
    name="heflwr",
    version="0.1.2",
    description="「HeFlwr」is a federated learning package for heterogeneous devices.",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    author="Zhu Yaolin",
    author_email="zhuyaolinluck@qq.com",
    long_description=readme(),
    long_description_content_type='text/x-rst',
    include_package_data=True,
    url="https://github.com/QVQZZZ/HeFlwr",
    license="MIT",
)
