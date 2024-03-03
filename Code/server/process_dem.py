import pickle
import numpy as np
from copy import deepcopy
from make_dem import plot_dem

def multi(a, b):
  # print(a[0][0])
  met = np.multiply(a,b)
  # print(a[0][0], met[0][0])
  # print(met)
  sum = 0
  for i in range(len(met)):
    for val in met[i]:
      sum+=val
  # print(sum , a/[0][0])
  return sum

def sub_array(a, i,j, conv):
  b = [[-1 for i in range(conv)] for j in range(conv)]
  for k in range(i,i+conv):
    for l in range(j, j+conv):
      b[k-i][l-j]=a[k][l]

  # print(b)
  return b

def find_conv(dem, conv= 5):
  val = 1/(1.0*conv*conv)
  m = [[1/(conv*conv) for i in range(conv)] for j in range(conv)]
  dem_temp = deepcopy(dem)
  for i in range(len(dem) - int(conv)):
    for j in range(len(dem[0]) - int(conv)):
      dem_temp[i+int(conv/2)][j+int(conv/2)] = multi(sub_array(dem, i, j, int(conv)), m)
  return dem_temp
  

def st_dev(dem, dem_temp):
  dem_standard = deepcopy(dem_temp)
  for i in range(len(dem)):
    for j in range(len(dem[0])):
      dem_standard[i][j] = (dem[i][j] - dem_temp[i][j])**2
  return dem_standard
  

def read_dem(file):
    file = open(file, 'r')
    dem = []
    for line in file.readlines():
      dem.append(line.split(" "))
    file.close()
    for i in range(len(dem)):
      dem[i] = dem[i][:-1]
    dem = dem[:][:-1]
    for i in range(len(dem)):
      for j in range(len(dem[0])):
        dem[i][j] = float(dem[i][j])

    return dem    

if __name__ == "__main__":
  file = "D:\\TataElxsi\\Code\\server\\DEM.txt"
  dem = read_dem(file)
  plot_dem(np.array(dem))

