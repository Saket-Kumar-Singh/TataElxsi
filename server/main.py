from map import map
from process_dem import read_dem
from vehicle import vehicle

if __name__ == "__main__":
    dem = read_dem("D:\TataElxsi\server\DEM.txt")
    obstacle = [[0 for i in range(len(dem[1]))] for j in range(len(dem))]
    map1 = map(dem, obstacle)
    v1 = vehicle(1)
