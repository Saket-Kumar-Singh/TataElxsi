from resources import math
from queue import PriorityQueue as PQ
from time import sleep
class vehicle:
  st = set([])
  def __init__(self, ID):
    self.ID = ID
    self.st.add(ID)

  def location(self, startingLocation, endLocation):
    self.stating_location = startingLocation
    self.end_location = endLocation

  def a_star(self, map, step_size = 5, alpha = 2):
    x = self.stating_location[0]
    y = self.stating_location[1]
    x1 = self.end_location[0]
    y1 = self.end_location[1]

    self.path = []
    self.vis = [[(-1, -1) for i in range(map.dim[1])] for j in range(map.dim[0])]
    # self.path.append((x, y))
    dir = [-1*step_size, 0, step_size]
    pq = PQ()
    pq.put((0,0, x, y, x, y))
    print(math.sqrt((x - x1)**2 + (y- y1)**2))
    self.vis[x][y] = (x, y)
    sleep(4)
    while (not pq.empty()) and (math.sqrt((x - x1)**2 + (y- y1)**2) > step_size*1.41):
      # print(x,y)
      tup = pq.get()
      hst, cst, x, y, x_o, y_o  = tup[0], tup[1], tup[2], tup[3], tup[4], tup[5]
      # print(x_o, y_o)
      self.vis[x][y] = (x_o, y_o)
      # self.path.append((x,y))
      for i in range(3):
        for j in range(3):
          if(x + dir[i] < map.dim[0] and x + dir[i] >= 0) and (y+ dir[j] < map.dim[1] and y + dir[j] >= 0):
            if self.vis[x + dir[i]][y + dir[j]] == (-1, -1):
              val = cst + map.cost(x, y, x + dir[i], y + dir[j]) + math.sqrt((step_size*dir[i])**2 + (step_size*dir[j])**2)
              val1 =  alpha*math.sqrt((x1 - x - dir[i])**2 + (y1 - y - dir[j])**2)
              pq.put((val+val1,val, x + dir[i] , y + dir[j], x, y))

    # print(x, y)
    (x1,y1) = self.stating_location
    lst = []
    lst.append(self.end_location)
    lst.append((x, y))
    # for i in range(10):
    #   for j in range(10):
    #     print(self.vis[i][j], end = " ")
      # print(end = "\n")  
    while(x != x1 or y != y1):
      (p,q)= self.vis[x][y]
      # print(p, q)
      x, y = p, q
      lst.append((x,y))

    return lst  
        # v[x][y] = count
        # count = count + 1       