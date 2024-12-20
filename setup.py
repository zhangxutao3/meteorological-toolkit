from setuptools import setup, find_packages

setup(
    name='meteorological_toolkit',  # 注意：包名最好不要有空格，可以用下划线
    version='0.1',
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",  # 修正拼写错误，如果是PyTorch，包名是torch
        "cupy",
        "pandas",
        "scipy",
    ],
    author='张栩滔',
    author_email='zxt0413363@163.com',
    description='气象数据处理工具箱',
    classifiers=[
        'Programming Language :: Python :: 3.10',
        'Operating System :: OS Independent',  # 改为 OS Independent
        'Operating System :: Linux',
        'Operating System :: Windows',
    ],
    python_requires='>=3.8',
)
