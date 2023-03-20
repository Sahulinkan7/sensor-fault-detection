from setuptools import setup,find_packages
from typing import List

PROJECT_NAME="sensor_fault_detection"
VERSION="0.0.1"
AUTHOR="Linkan Kumar Sahu"
AUTHOR_EMAIL="sahulinkan7@gmail.com"
PACKAGES=find_packages()
DESCRIPTION="A machine learning project on sensorfault detection"

REQUIREMENT_FILE="requirements.txt"

def get_requirements()->List[str]:
    '''
    This function will retun the list of requirements 
    mentioned in requirements.txt file
    
    '''
    with open(REQUIREMENT_FILE) as file:
        requirement_list=file.readlines()
        requirement_list=[item.replace("\n","") for item in requirement_list]
        requirement_list.remove("-e .")
    return requirement_list


setup(name=PROJECT_NAME,
      version=VERSION,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      packages=PACKAGES,
      description=DESCRIPTION,
      install_requires=get_requirements())