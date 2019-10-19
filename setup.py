# -*- coding:utf-8 -*-
from setuptools import setup,find_packages

setup(name='edakit', 
      version='0.1', 
      packages=find_packages(),
      include_package_data=True,   # 启用清单文件MANIFEST.in
      exclude_package_data={'':['.gitignore','.git']}, #排除文件列表      
      install_requires=[           # 依赖列表
        'xgboost>=0.80'
      ],
      long_description=__doc__,
      url='https://github.com/lyhue1991/tianjikit', 
      author='Python_Ai_Road', 
      author_email='lyhue1991@163.com', 
      zip_safe=False
     )