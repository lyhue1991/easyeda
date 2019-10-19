# -*- coding:utf-8 -*-
from setuptools import setup,find_packages

setup(name='edatools', 
      version='1.0', 
      packages=find_packages(),
      include_package_data=True,   # 启用清单文件MANIFEST.in
      exclude_package_data={'':['.gitignore','.git']}, #排除文件列表      
      install_requires=[           # 依赖列表
        'pandas>=0.24.2',
        'tqdm>=4.31.1'
      ],
      long_description=__doc__,
      url='https://github.com/lyhue1991/edatools', 
      author='Python_Ai_Road', 
      author_email='lyhue1991@163.com', 
      zip_safe=False
     )
