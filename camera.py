# read https://www.cs.cmu.edu/~16385/s17/Slides/11.1_Camera_matrix.pdf
import numpy as np

# note camera and world coordinate system are the same
# camera matrix
# focal length = 1 (fx, fy) = (1, 1)
# principal point = (cx, cy) = (0, 0)

M = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
r = M @ np.array([1600, 1200, 5])
print(r[0] / r[2], r[1] / r[2])  # (320, 200)

# we move the camera in the z direction by 1
r = M @ np.array([1600, 1200, 5 - 1])
print(r[0] / r[2], r[1] / r[2])  # (400, 300)
