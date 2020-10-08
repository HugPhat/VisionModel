from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="visTorch", # Replace with your own username
    version="0.0.1",
    author="Hug Phat",
    author_email="hug.phat.vo@gmail.com",
    description="DeepLearning Model in Pytorch",
    long_description=long_description,
    long_description_content_type="",
    url="",
    packages=find_packages(where='./Backbone') + 
                find_packages(where='./Dataset') + 
                find_packages(where='./Head')
                ,
    install_requires=[
       
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)