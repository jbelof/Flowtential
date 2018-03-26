import pandas as pd
from keras.models import load_model, model_from_json
import sys
import keras.backend as K
import os
from numba.decorators import jit
import numpy as np
import simplejson
from sklearn.preprocessing import MinMaxScaler
import math
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# sets the GPU to invisible if empty string, otherwise specify the GPU id
os.environ['CUDA_VISIBLE_DEVICES']=''

# path to the model definition file which is *.json
modelpath = sys.argv[1]

# path to the weights file which is *.h5
weightspath = sys.argv[2]

# path to the coordinate file
coordspath = sys.argv[3]

coords = pd.read_csv(coordspath, header=None).as_matrix()
coords = coords.reshape((1, coords.shape[0]*3))
mm = MinMaxScaler((0.1,0.9)).fit([-200,200])

json_file = open(modelpath, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(weightspath)
loaded_model.compile(optimizer='rmsprop', loss='mean_absolute_percentage_error')

def get_distances(yy):
  dist = np.zeros((yy.shape[0], yy.shape[1]/3 * (yy.shape[1]/3 - 1) / 2))
  for i in range(yy.shape[0]):
    ctr = 0
    for j in range(yy.shape[1]/3):
      ax = yy[i, j*3]
      ay = yy[i, j*3+1]
      az = yy[i, j*3+2]
      for k in range(j+1, yy.shape[1]/3):
        bx = yy[i,k*3]
        by = yy[i,k*3+1]
        bz = yy[i,k*3+2]
        tmpx = ax-bx
        tmpy = ay-by
        tmpz = az-bz
	boxhalf = 7.5
        if(tmpx < -boxhalf): tmpx = tmpx + boxhalf
        if(tmpx > boxhalf): tmpx = tmpx - boxhalf
        if(tmpy < -boxhalf): tmpy = tmpy + boxhalf
        if(tmpy > boxhalf): tmpy = tmpy - boxhalf
        if(tmpz < -boxhalf): tmpz = tmpz + boxhalf
        if(tmpz > boxhalf): tmpz = tmpz - boxhalf

        dist[i,ctr] = math.sqrt(tmpx*tmpx + tmpy*tmpy + tmpz*tmpz)
        ctr += 1
  return dist

jitdist = jit(get_distances)
preds = loaded_model.predict(jitdist(coords)/27)
preds = mm.inverse_transform(preds)
preds = preds.reshape((108,3))
np.savetxt('./dnn.forces.dat', preds, delimiter=' ')

