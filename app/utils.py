from resources import np, deepcopy, copy
import math

def multi(a, b):
  met = np.multiply(a,b)
  sum = 0
  for i in range(len(met)):
    for val in met[i]:
      sum+=val
  return sum

def sub_array(a, i,j, conv):
  b = [[-1 for i in range(conv)] for j in range(conv)]
  for k in range(i,i+conv):
    for l in range(j, j+conv):
      b[k-i][l-j]=a[k][l]
  return b

# Convert the values of dem to numbers as the input is in string

from copy import copy, deepcopy

def find_conv(dem, conv= 20):
  val = 1/(1.0*conv*conv)
  m = [[1/(conv*conv) for i in range(conv)] for j in range(conv)]
  dem_temp = deepcopy(dem)
  for i in range(len(dem) - int(conv)):
    for j in range(len(dem[0]) - int(conv)):
      dem_temp[i+int(conv/2)][j+int(conv/2)] = multi(sub_array(dem, i, j, int(conv)), m)
  return dem_temp    
  plot_dem(np.array(dem_temp))


def st_dev(dem, dem_temp):
  dem_standard = deepcopy(dem_temp)
  for i in range(len(dem)):
    for j in range(len(dem[0])):
      dem_standard[i][j] = (dem[i][j] - dem_temp[i][j])**2
  return dem_standard
  plot_dem(np.array(dem_standard))

def abs(x):
   if x > 0:
      return x
   return -1*x   

def M(theta):
    """
    Return the angle phi = theta mod (2 pi) such that -pi <= theta < pi.
    """
    theta = theta % (2*math.pi)
    if theta < -math.pi: return theta + 2*math.pi
    if theta >= math.pi: return theta - 2*math.pi
    return theta

def R(x, y):
    """
    Return the polar coordinates (r, theta) of the point (x, y).
    """
    r = math.sqrt(x*x + y*y)
    theta = math.atan2(y, x)
    return r, theta

def change_of_basis(p1, p2):
    """
    Given p1 = (x1, y1, theta1) and p2 = (x2, y2, theta2) represented in a
    coordinate system with origin (0, 0) and rotation 0 (in degrees), return
    the position and rotation of p2 in the coordinate system which origin
    (x1, y1) and rotation theta1.
    """
    theta1 = deg2rad(p1[2])
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    new_x = dx * math.cos(theta1) + dy * math.sin(theta1)
    new_y = -dx * math.sin(theta1) + dy * math.cos(theta1)
    new_theta = p2[2] - p1[2]
    return new_x, new_y, new_theta

def rad2deg(rad):
    return 180 * rad / math.pi

def deg2rad(deg):
    return math.pi * deg / 180

def sign(x):
    return 1 if x >= 0 else -1