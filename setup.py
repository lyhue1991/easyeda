# -*- coding:utf-8 -*-
#from setuptools import setup,find_packages
# setup(name='edatools', 
#       version='1.0', 
#       packages=find_packages(),
#       include_package_data=True,   # 启用清单文件MANIFEST.in
#       exclude_package_data={'':['.gitignore','.git']}, #排除文件列表      
#       install_requires=[           # 依赖列表
#         'pandas>=0.24.2',
#         'tqdm>=4.31.1'
#       ],
#       long_description=__doc__,
#       url='https://github.com/lyhue1991/edatools', 
#       author='Python_Ai_Road', 
#       author_email='lyhue1991@163.com', 
#       zip_safe=False
#      )

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="easyeda",
    version="2.0",
    author="PythonAiRoad",
    author_email="lyhue1991@163.com",
    description="A useful tool to Exploratory Data Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lyhue1991/easyeda",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5'
)

