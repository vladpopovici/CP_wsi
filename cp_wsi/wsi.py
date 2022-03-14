#   -*- coding: utf-8 -*-
#
#  --------------------------------------------------------------------
#  Copyright (c) 2022 Vlad Popovici <popovici@bioxlab.org>
#
#  Licensed under the MIT License. See LICENSE file in root folder.
#  --------------------------------------------------------------------

__author__ = "Vlad Popovici <popovici@bioxlab.org>"
__version__ = "0.3.5"

"""
WSI: Defines WSI class.
"""

__all__ = ['WSIInfo', 'NumpyImage']

import zarr
import pathlib
import numpy as np
from math import log

from .magnif import Magnification
from typing import NewType, Optional

ImageShape = NewType("ImageShape", dict[str, int])


#####
class WSIInfo(object):
    """Hold some basic info about a WSI.

    Args:
        path (str): full path to root of imported WSI file. This is the folder
            containing a 'pyramid' ZARR group with datasets for each level. Normally,
            for an imported slide <SLIDE_NAME>, the canonical path should be
            .../<SLIDE_NAME>/slide/pyramid.zarr.

    Attributes:
        path (str): full path to WSI file
        info (dict): a dictionary containing WSI properties
            Example:
            {'background': 'FFFFFF',
             'objective_power': 20,
             'pyramid': [{'downsample_factor': 1, 'height': 111388, 'level': 0, 'width': 75829},
                         {'downsample_factor': 2, 'height': 55694, 'level': 1, 'width': 37914},
                         {'downsample_factor': 4, 'height': 27847, 'level': 2, 'width': 18957},
                         {'downsample_factor': 8, 'height': 13923, 'level': 3, 'width': 9478},
                         {'downsample_factor': 16, 'height': 6961, 'level': 4, 'width': 4739},
                         {'downsample_factor': 32, 'height': 3480, 'level': 5, 'width': 2369},
                         {'downsample_factor': 64, 'height': 1740, 'level': 6, 'width': 1184},
                         {'downsample_factor': 128, 'height': 870, 'level': 7, 'width': 592},
                         {'downsample_factor': 256, 'height': 435, 'level': 8, 'width': 296},
                         {'downsample_factor': 512, 'height': 217, 'level': 9, 'width': 148}],
            'resolution_units': 'microns',
            'resolution_x_level_0': 0.23387573964497,
            'resolution_y_level_0': 0.234330708661417, 'vendor': 'mirax'}
    """

    def __init__(self, path: str):
        self.info = {}  # Info
        self._pyramid_levels = None  # Convenient access to pyramid levels [0,]->width, [1,]->height
        self.path = pathlib.Path(path)

        with zarr.open(self.path, mode='r') as z:
            self.info = z.attrs['metadata']
            self.info['pyramid'] = z.attrs['pyramid']

        if len(self.info['pyramid']) > 1:
            ms = self.info['pyramid'][1]['downsample_factor']/self.info['pyramid'][0]['downsample_factor']
        else:
            ms = 1.0
        self.magnif_converter =  Magnification(self.info['objective_power'],
                                               mpp=0.5*(self.info['resolution_x_level_0']+self.info['resolution_y_level_0']),
                                               level=0,
                                               magnif_step=ms)

        self._pyramid_levels = [{'width': p['width'], 'height': p['height']} for p in self.info['pyramid']]
        return


    def level_count(self) -> int:
        """Return the number of levels in the multi-resolution pyramid."""
        return len(self._pyramid_levels)


    def downsample_factor(self, level:int) -> int:
        """Return the downsampling factor (relative to level 0) for a given level."""
        if level < 0 or level >= self.level_count():
            return -1
        for p in self.info['pyramid']:
            if p['level'] == level:
                return p['downsample_factor']


    def get_native_magnification(self) -> float:
        """Return the original magnification for the scan."""
        return self.info['objective_power']


    def get_native_resolution(self) -> float:
        """Return the scan resolution (microns per pixel)."""
        return 0.5 * (self.info['resolution_x_level_0'] + self.info['resolution_y_level_0'])


    def get_level_for_magnification(self, mag: float, eps=1e-6) -> int:
        """Returns the level in the image pyramid that corresponds the given magnification.

        Args:
            mag (float): magnification
            eps (float): accepted error when approximating the level

        Returns:
            level (int) or -1 if no suitable level was found
        """
        if mag > self.info['objective_power'] or mag < 2.0**(1-self.level_count()) * self.info['objective_power']:
            return -1

        #lx = log2(self.info['objective_power'] / mag)
        lx = log(self.info['objective_power'] / mag, self.magnif_converter._magnif_step)
        k = np.where(np.isclose(lx, range(0, self.level_count()), atol=eps))[0]
        if len(k) > 0:
            return k[0]   # first index matching
        else:
            return -1   # no match close enough


    def get_level_for_mpp(self, mpp: float):
        """Return the level in the image pyramid that corresponds to a given resolution."""
        return self.magnif_converter.get_level_for_mpp(mpp)


    def get_mpp_for_level(self, level: int):
        """Return resolotion (mpp) for a given level in pyramid."""
        return self.magnif_converter.get_mpp_for_level(level)


    def get_magnification_for_level(self, level: int) -> float:
        """Returns the magnification (objective power) for a given level.

        Args:
            level (int): level in the pyramidal image

        Returns:
            magnification (float)
            If the level is out of bounds, returns -1.0
        """
        if level < 0 or level >= self.level_count():
            return -1.0
        if level == 0:
            return self.info['objective_power']

        #return 2.0**(-level) * self.info['objective_power']
        return self.magnif_converter._magnif_step ** (-level) * self.info['objective_power']


    def get_extent_at_level(self, level: int) -> Optional[ImageShape]:
        """Returns width and height of the image at a desired level.

        Args:
            level (int): level in the pyramidal image

        Returns:
            (width, height) of the level
        """
        if level < 0 or level >= self.level_count():
            return None
        return self._pyramid_levels[level]


#####
class NumpyImage:
    """This is merely a namespace for collecting a number of useful
    functions that are applied to images stored as Numpy arrays.
    Usually, such an image -either single channel or 3(4) channels -
    is stored as a H x W (x C) array, with H (height) rows and W (width)
    columns. C=3 or 4.
    """

    @staticmethod
    def width(img):
        img: np.ndarray
        return img.shape[1]

    @staticmethod
    def height(img):
        img: np.ndarray
        return img.shape[0]

    @staticmethod
    def nchannels(img):
        img: np.ndarray
        if img.ndim > 2:
            return img.shape[2]
        else:
            return 1

    @staticmethod
    def is_empty(img, empty_level: float=0) -> bool:
        """Is the image empty?

        Args:
            img (numpy.ndarray): image
            empty_level (int/numeric): if the sum of pixels is at most this
                value, the image is considered empty.

        Returns:
            bool
        """

        return img.sum() <= empty_level

    @staticmethod
    def is_almost_white(img, almost_white_level: float=254, max_stddev: float=1.5) -> bool:
        """Is the image almost white?

        Args:
            img (numpy.ndarray): image
            almost_white_level (int/numeric): if the average intensity per channel
                is above the given level, decide "almost white" image.
            max_stddev (float): max standard deviation for considering the image
                almost constant.

        Returns:
            bool
        """

        return (img.mean() >= almost_white_level) and (img.std() <= max_stddev)
