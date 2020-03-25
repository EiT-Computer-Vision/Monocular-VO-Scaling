import numpy as np
def rotateX(theta):
	ang = theta * np.pi / 180
	c = np.cos(ang)
	s = np.sin(ang)

	a = np.identity(4)
	rot = np.array([[c, -s], [s, c]])
	a[1:3, 1:3] = rot
	return a


def rotateY(theta):
	ang = theta * np.pi / 180
	c = np.cos(ang)
	s = np.sin(ang)
	a = np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])
	return a


def rotateZ(theta):
	ang = theta * np.pi / 180
	c = np.cos(ang)
	s = np.sin(ang)
	a = np.identity(4)
	rot = np.array([[c, -s], [s, c]])
	a[0:2, 0:2] = rot
	return a