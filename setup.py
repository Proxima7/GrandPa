from distutils.core import setup

install_requires = [
    "setuptools",
    "wheel",
    "imageio",
    "numpy",
    "opencv-python",
    "Pillow",
    "psutil",
    "tqdm",
]


setup(
    name='grandpa',
    version='0.6.2',
    packages=['grandpa', 'grandpa.utils'],
    url='https://github.com/Proxima7/GrandPa',
    license='MIT License',
    author='Bizerba AI Team',
    author_email='pascal.iwohn@bizerba.com',
    description='',
    install_requires=install_requires,
    package_dir={"": "src"},
)
