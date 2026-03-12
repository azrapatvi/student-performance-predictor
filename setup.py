# with the help of setup.py you project becomes installable like library

from setuptools import find_packages,setup

def get_requirements(file_name):
    requirements=[]
    with open(file_name) as f:
        requirements = f.read().split('\n')

    if '-e .' in requirements:
        requirements.remove('-e .')
        
    return requirements


setup(
    name='mlproject',
    version='0.0.1',
    author='azra',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')

)