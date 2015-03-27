'''
Basic objects for LGK model description.
'''
import matrix
import math
import numpy as np
from collections import OrderedDict
from itertools import product

COLLIMATOR_SIZES = ( 4, 8, 16 )

# Number of sectors
SECTOR_COUNT = 8

# Number of rings
RING_COUNT = 5

# How many collimator channels are there in each ring
RING_MULTIPLICITY = OrderedDict(((1, 6), (2, 4), (3, 5), (4, 4), (5, 5)))

COLLIMATORS_PER_SECTOR = 24

COLLIMATOR_COUNT = COLLIMATORS_PER_SECTOR * SECTOR_COUNT

COLLIMATORS = tuple(product(COLLIMATOR_SIZES, RING_MULTIPLICITY.keys()))

LGP_RING_OFS = {
    "tmr10" : OrderedDict((
        ((4, 1), 0.812),
        ((4, 2), 0.823),
        ((4, 3), 0.795),
        ((4, 4), 0.726),
        ((4, 5), 0.664),
        ((8, 1), 0.934),
        ((8, 2), 0.919),
        ((8, 3), 0.874),
        ((8, 4), 0.782),
        ((8, 5), 0.708),
        ((16, 1), 0.961),
        ((16, 2), 1.000),
        ((16, 3), 0.981),
        ((16, 4), 0.914),
        ((16, 5), 0.847),
    )),
    "classic" : OrderedDict((
        ((4, 1), 0.799),
        ((4, 2), 0.815),
        ((4, 3), 0.792),
        ((4, 4), 0.725),
        ((4, 5), 0.663),
        ((8, 1), 0.957),
        ((8, 2), 0.946),
        ((8, 3), 0.901),
        ((8, 4), 0.808),
        ((8, 5), 0.730),
        ((16, 1), 0.961),
        ((16, 2), 1.000),
        ((16, 3), 0.986),
        ((16, 4), 0.920),
        ((16, 5), 0.851)
    ))
}

LGP_EFF_OFS = {
    "tmr10" : OrderedDict((
        (4, 0.814),
        (8, 0.900),
        (16, 1.000)
    )),
    "classic" : OrderedDict((
        (4, 0.805),
        (8, 0.914),
        (16, 1.000)
    ))
}

class Shot(object):
    def __init__(self, isocentre=(100., 100., 100.), sectors=4, angle=90, weight=1.0):
        self.isocentre = np.array(isocentre) # X, Y, Z (in mm), as in LGP
        self.sectors = sectors               # one or eight [4|8|16|B]
        self.weight = weight                 # currently not used
        self.angle = angle                   # gamma angle in deg

    @property
    def rotation_matrix(self):
        '''Matrix for rotation from stereotactic to machine coordinates.'''
        angle = (90 + self.angle) * math.pi / 180.
        return matrix.rotate_x(angle)

    @property
    def inverse_matrix(self):
        '''Matrix for rotation from machine to stereotactic coordinates.'''
        angle = (90 + self.angle) * math.pi / 180.
        return matrix.rotate_x(-angle)

    @property
    def is_centered(self):
        return self.isocentre == (100., 100., 100.)

    def transform(self, stereotactic):
        '''Transform from stereotactic to machine coordinates.'''
        # print stereotactic.shape, stereotactic.dtype
        # print self.isocentre.shape, self.isocentre.dtype
        delta = stereotactic - self.isocentre
        return self.rotation_matrix.dot(delta)

    def inverse_transform(self, machine):
        '''Transform from machine to stereotactic coordinates.'''
        return self.matrix.inverse_matrix.dot(machine) + self.isocentre


class PhantomRegistration(object):
    '''Registration of DICOM in stereotactic frame.'''
    
    def __init__(self):
        # Vectors of X,Y,Z in DICOM coordinates. Needn't be normalized
        self.vec_x = None
        self.vec_y = None
        self.vec_z = None

        self.center = None    # Central point (100, 100, 100) in DICOM coordinates

    @property
    def rotation_matrix(self):
        '''Rotation ijk to stereotactic.'''
        return matrix.from_vectors(self.vec_x, self.vec_y, self.vec_z).transpose() 
        # Inverse equals transposed for rotation matrix

    def to_stereotactic(self, vec):
        '''Transform vector from ijk to stereotactic.'''
        return np.array([100, 100, 100]) + self.rotation_matrix.dot(np.array(vec) - np.array(self.center))

    def from_stereotactic(self, vec):
        vec = np.array(vec)
        broadcast_shape = (3,) + (len(vec.shape) - 1) * (1,)
        translation = vec - np.array([100, 100, 100]).reshape(broadcast_shape)
        return self.rotation_matrix.T.reshape((3,) + broadcast_shape).dot(translation) + np.array(self.center).reshape(broadcast_shape)

class Target(object):
    def __init__(self, center, grid):
        self.center = np.array(center)
        self.grid = grid

    def to_stereotactic(self, indices):
        '''Convert machine 
        '''
        # print indices
        indices = np.array(indices)
        broadcast_shape = (3,) + (len(indices.shape) - 1) * (1,)
        return self.center.reshape(broadcast_shape) + (indices - 15) * self.grid