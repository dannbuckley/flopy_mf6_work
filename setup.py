from setuptools import setup, find_packages

setup(
    packages=find_packages(
        where="src",
        include=["flopy_mf6_work*"],
    ),
    package_dir={"": "src"},
)
