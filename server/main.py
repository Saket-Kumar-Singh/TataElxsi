from map import map
from process_dem import read_dem
from vehicle import vehicle
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib import cbook, cm
import numpy as np

if __name__ == "__main__":
    dem = read_dem("D:\\TataElxsi\\server\\DEM.txt")
    obstacle = [[0 for i in range(len(dem[1]))] for j in range(len(dem))]
    map1 = map(dem, obstacle)
    v1 = vehicle(1)
    v1.location((0, 0), (50, 50))
    lst = v1.a_star(map1)
    print(lst)

    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ls = LightSource(270, 45)
    region = np.s_[5:50, 5:50]
    x = np.linspace(0, 50, 50)
    y = np.linspace(0, 50, 50)
    z = []
    # x = []
    # y  = []
    for i in range(50):
      # for j in range(50):
        # x.append(i)
        # y.append(j)
        # z.append([0 for i in range(50)])
        z.append(map1.obstacle[i][:50])


    z = np.array(z)
    ax.plot_surface(x, y, z, cmap="RdYlGn", linewidth=1, alpha = 0.35)
    z = np.array(z)
    # z = np.array(map1.dem[:50])

    # print(z.shape)



    # rgb = ls.shade(z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
    # surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb,
    #                        linewidth=0, antialiased=False, shade=False)

    # p = plot_dem(z)

    # for pos in lst:
    x = []
    y = []
    z = []
    z_ = []
    for pos in lst:
      x.append(pos[0])
      y.append(pos[1])
      z.append(map1.obstacle[pos[0]][pos[1]] + 0.5)
      z_.append(-2)
      # z.append( 0)

    ax.scatter3D(x, y, z , color = "red", s = 10)
    ax.plot(x, y, z , color = 'black', linewidth = 3)   
    plt.show() 