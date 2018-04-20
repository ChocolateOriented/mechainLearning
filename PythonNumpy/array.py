import numpy as np

randMat = np.mat(np.random.rand(4,4));
invRandMat = randMat.I
print(randMat * invRandMat)

np.tile()