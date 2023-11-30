from distutils.core import setup

install_requires = [
    "dill~=0.3.7",
]


setup(
    name='grandpa',
    version='2.0.0',
    packages=['grandpa'],
    url='https://github.com/Proxima7/GrandPa',
    license='MIT License',
    author='Bizerba AI Team',
    author_email='pascal.iwohn@bizerba.com',
    description='',
    install_requires=install_requires,
    package_dir={"": "src"},
)
