"""Functions for creation and manipulation with rotational matrices."""
from math import atan2, asin, cos, sin, pi, sqrt
import numpy as np

def normalize(v):
    '''Normalize a vector.

    Taken from http://stackoverflow.com/questions/21030391/how-to-normalize-array-numpy
    '''
    v = np.array(v)
    norm = sqrt(v.dot(v))
    if norm == 0: 
       raise Exception("Cannot normalize zero-length vector.")
    return v / norm

# Rotations http://stackoverflow.com/questions/15022630/how-to-calculate-the-angle-from-roational-matrix
# http://www.soi.city.ac.uk/~sbbh653/publications/euler.pdf
def rotate_x(angle):
    return np.array([[1,0,0], [0, cos(angle), -sin(angle)], [0, sin(angle), cos(angle)]])

def rotate_y(angle):
    return np.array([[cos(angle), 0, sin(angle)], [0, 1, 0], [-sin(angle), 0, cos(angle)]])

def rotate_z(angle):
    return np.array([[cos(angle), -sin(angle), 0], [sin(angle), cos(angle), 0], [0,0,1]])

def rotation_matrix(z=0.0, y=0.0, x=0.0):
    '''Rotation matrix from three rotation angles.

    Returns R_z(z) . R_y(y) . R_x(x)'''
    return rotate_z(z).dot(rotate_y(y)).dot(rotate_x(x))

def from_vectors(i_xyz, j_xyz, k_xyz):
    '''Rotation matrix for transition from coords ijk to coords xyz.

    :param i_xyz: axes i in x,y,z coors
    :param j_xyz: axes j in x,y,z coors
    :param k_xyz: axes k in x,y,z coors

    Vectors don't have to be normalized.
    '''
    matrix = np.ndarray((3, 3))
    matrix[:,0] = normalize(i_xyz)
    matrix[:,1] = normalize(j_xyz)
    matrix[:,2] = normalize(k_xyz)

    # TODO: Check that vectors are orthogonal

    return matrix

def get_angles(m):
    '''Rz(phi) * Ry(theta) * Rx(psi).

    Returns psi, theta, phi (in radians)
    Usually, there are at least two solutions, we choose one of them.
    '''

    def normalize(angle):
        '''Angles very close to zero have no real meaning in our application.'''
        if abs(angle) < 1e-10:
            return 0.0
        else:
            return angle

    if abs(m[2,0]) in [-1.0, 1]:
        phi = 0
        if m[2, 0] == -1:
            theta = pi / 2
            psi = phi + atan2(m[0,1], m[0, 2])
        else:
            theta = -pi / 2
            psi = phi + atan2(-m[0,1], -m[0, 2])
    else:
        theta1 = -asin(m[2, 0])
        theta2 = pi - theta1

        psi1 = atan2(m[2,1] / cos(theta1), m[2,2] / cos(theta1))
        psi2 = atan2(m[2,1] / cos(theta2), m[2,2] / cos(theta2))

        phi1 = atan2(m[1,0] / cos(theta1), m[0,0] / cos(theta1))
        phi2 = atan2(m[1,0] / cos(theta2), m[0,0] / cos(theta2))

        theta = theta1
        psi = psi1
        phi = phi1

    return np.array([normalize(psi), normalize(theta), normalize(phi)])

if __name__ == "__main__":
    import numpy as np

    one = - np.identity(3)

    second = np.ndarray((3, 3)) 
    second[0,0] = 1
    second[1,1] = -1
    second[2,2] = -1

    # print get_angles(one)
    # print get_angles(second)

    m = rotate_z(30 * pi / 180) * 5
    print m
    m2 = from_vectors(m[:,0], m[:,1], m[:,2])
    print m2

