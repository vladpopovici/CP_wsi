#   -*- coding: utf-8 -*-
#
#  --------------------------------------------------------------------
#  Copyright (c) 2022 Vlad Popovici <popovici@bioxlab.org>
#
#  Licensed under the MIT License. See LICENSE file in root folder.
#  --------------------------------------------------------------------

from setuptools import setup

setup(
    name='cp_wsi',
    version='0.3.5',
    description='Whole Slide Image class and tools.',
    url='https://github.com/vladpopovici/CP_wsi',
    author='Vlad Popovici',
    author_email='popovici@bioxlab.org',
    license='MIT',
    packages=['cp_wsi'],
    install_requires=['numpy',
                      'zarr',
                      ],

    classifiers=[
        'Development Status :: 4 - Beta'
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8'
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
