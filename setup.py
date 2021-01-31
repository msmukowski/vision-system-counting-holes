"""A setuptools based setup module.
"""
from setuptools import setup, find_packages

# Long description from the README.md file
with open(file="README.md", mode="r") as readme_handle:
    long_description = readme_handle.read()

setup(
    name='vision-system-holes-counter',
    version='1.0.0',
    description='Vision system to recognize objects in an image and count the holes in each of them.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/msmukowski/vision-system-counting-holes',
    author='Maciej Smukowski (msmukowski)',
    author_email='msmukowski@gmail.com',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Education',
        'Topic :: Scientific/Engineering :: Image Processing',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8'
    ],
    keywords='image, processing, vision, system, recognition',
    packages=find_packages(where='vsch'),
    package_dir={'','vsch'},
    python_requires='>=3.6, <4',
    entry_points={
        'console_scripts': [
            'run-vshc = vsch.main:main'
        ]
    }

)