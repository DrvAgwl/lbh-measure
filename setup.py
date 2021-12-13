from setuptools import setup, find_packages

REQUIRED_PACKAGES = list(filter(None, list(map(lambda s: s.strip(), open('requirements.txt').readlines()))))

with open("README.md", "r") as readme:
    long_description = readme.read()
setup(
    name='lbh-measure',
    version='1.0',
    author="Nikhil Kasukurthi",
    author_email="nikhil.k@udaan.com",
    description="SDK for LBH Measure, training and deployment",
    long_description_content_type="text/markdown",
    url="https://github.com/udaan-com/lbh-measure-python-service",
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    # packages=find_packages(),
    packages=find_packages(include=['lbh_measure', 'lbh_measure.*']),
    classifiers=[
        'Programming Language :: Python :: 2.7'
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
