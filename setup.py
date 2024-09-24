from setuptools import find_packages,setup

HYPEN_E_DOT='-e .'

def get_requirements(file_path):
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

        return requirements

if __name__=="__main__":
    setup(
        name='ProsePredict: Advanced LSTM & GRU Text Forecasting',
        version='1.1',
        author='Bhuvan Kapoor',
        author_email='bhuvankapoor123@gmail.com',
        install_requires=get_requirements('requirements.txt'),
        packages=find_packages()

    )