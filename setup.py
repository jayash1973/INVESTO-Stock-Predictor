from setuptools import find_packages, setup
from typing import List


HYPEN_E_DOT='-e .'

def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list of requirements from the file
    '''
    requirements=[]
    with open(file_path,'r') as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]
        
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
            
    return requirements

setup(
    name='INVESTO-Stock-Prediction',
    version='0.0.1',
    author='Jayash Bhardwaj',
    author_email='jayashbhardwaj294@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    python_requires=">=3.8",
    description="A stock prediction application",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)