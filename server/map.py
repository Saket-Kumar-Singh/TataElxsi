from process_dem import  find_conv, st_dev
import pickle
from queue import PriorityQueue as PQ
from resources import np, noise, gaussian_filter, math, PQ

class map:
  def __init__(self, dem, obstacle, width = 6):
    self.dem = dem
    self.obstacle = obstacle
    self.dem_temp = find_conv(dem, conv = 5)
    self.dem_st_dev = st_dev(dem, self.dem_temp)
    self.add_obstacles()
    self.dim = np.shape(np.array(self.obstacle))
    self.bfs()
    self.path = np.array([[-1 for i in range(self.dim[1])] for j in range(self.dim[0])])
    print(self.path.shape)
    self.make_penalty(width)

  def add_obstacles(self):
    for i in range(len(self.dem)):
      for j in range(len(self.dem[0])):
        if self.dem_st_dev[i][j] > 4 :
          self.obstacle[i][j] = 1

  def bfs(self):
    lst = []
    for i in range(self.dim[0]):
      for j in range(self.dim[1]):
        if (self.obstacle[i][j] == 1):
          lst.append((i,j))
    dir = [-1,0,1]
    # dir = [-1,0,1]
    while len(lst):
      p = lst.pop(0)
      for i in range(3):
        for j in range(3):
          try:
            if (p[0] + dir[i] < self.dim[0]) and (p[1]+dir[j] < self.dim[1]) and (p[0] + dir[i] > 0) and (p[1] + dir[j] > 0) and ((self.obstacle[p[0]+dir[i]][p[1]+dir[j]] == 0) or (self.obstacle[p[0]+dir[i]][p[1]+dir[j]] > self.obstacle[p[0]][p[1]] + 1)):
              self.obstacle[p[0]+dir[i]][p[1]+dir[j]] = self.obstacle[p[0]][p[1]] + 1
              lst.append((p[0]+dir[i],p[1]+dir[j]))
          except :
            print("error at ", p[1]+dir[j], p[0] + dir[i], self.dim[0],  self.dim[1])

  def func(self, x, width):
    if(x <= width):
      return 100
    else:
      return 100*math.e**(-1*((x-width)**2))

  def make_penalty(self, width):
    # print(self.dim)
    for i in range(self.dim[0]):
      for j in range(self.dim[1]):
        try:
          self.path[i][j] = self.func(self.obstacle[i][j], width)
        except:
          try:
            self.path[i][j] = -1
            print("Error with obstaCLE")
          except:
            print("Path size small")



  def cost(self, x, y, x1, y1):
    val = 0.0
    for i in range(x, x1+1):
      for j in range(y, y1+1):
        val += self.dem[i][j]
    try:
      val = val/(abs((x-x1 + 1)*(y - y1 + 1)))
    except :
      val = val
    ans = 0.0
    for i in range(min(x, x1), max(x, x1) + 1):
      for j in range(min(y, y1), max(y, y1)+1):
        ans+=((self.dem[i][j] - val)**2)
    try:
      return ans/(abs((x1 - x + 1)*(y1 - y  + 1)))
    except:
      return ans