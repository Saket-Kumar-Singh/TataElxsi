import numpy as np
import matplotlib.pyplot as plt
import noise
from scipy.ndimage import gaussian_filter
import os

if os.path.exists('DEM.txt'):
    print('The file exists!')
else:
    print("This doesn't exist")

def plot_dem(dem):
    fig = plt.figure()
    dem_temp = dem
    ax = fig.add_subplot(111, projection='3d')
    rows, cols = np.array(dem).shape
    x = np.arange(0, cols, 1)
    y = np.arange(0, rows, 1)
    x, y = np.meshgrid(x, y)
    ax.plot_surface(x, y, dem, cmap='terrain')
    plt.show()


class rand_dem:
    def __intit__(self, file = "Rand_DEM.txt", rows = 200, columns = 200, scale=100.0, octaves=0.3, persistence=0.5, lacunarity=0.0, seed=None, smoothness=3.0, new_min=0, new_max=100, make_file = True):
        self.dem = self.generate_smooth_demenerate(
            rows, columns, scale, octaves, persistence, 
            lacunarity, seed, smoothness, new_min, new_max)
        if(self.make_file):
            self.write()

    def generate_smooth_dem(rows, cols, scale=100.0, octaves=0.3, persistence=0.5, lacunarity=0.0, seed=None, smoothness=3.0, new_min=0, new_max=100):
        world = np.zeros((rows, cols))

        for i in range(rows):
            for j in range(cols):
                world[i][j] = noise.pnoise2(i/scale, j/scale, octaves=octaves, persistence=persistence, lacunarity=lacunarity, repeatx=1024, repeaty=1024, base=seed)

        min_value = np.min(world)
        max_value = np.max(world)
        world = (world - min_value) / (max_value - min_value)  # Normalize to range [0, 1]

        # Apply Gaussian filter for increased smoothness
        world_smooth = gaussian_filter(world, sigma=smoothness)

        # Scale to the new range [new_min, new_max]
        world_smooth_scaled = (world_smooth * (new_max - new_min)) + new_min

        return world_smooth_scaled
        # return world_smooth_scaled
    
    def write(self):
        try:
            file = open(self.file, 'a')
            for i in range(len(self.dem)):
              for j in range(len(self.dem[0])):
                file.write(str(self.dem[i][j]))
                file.write(" ")
              file.write("\n")
            file.close()
        except:
            print(f"No file named {file} exsist")

def read_dem(file):
    file = open(file, 'r')
    dem = []
    for line in file.readlines():
        dem.append(line.split(" "))
    file.close()
    for i in range(len(dem)):
        for j in range(len(dem[0])):
          dem[i][j] = float(dem[i][j])
    for i in range(len(dem)):
        dem[i] = dem[i][:-1]
    dem = dem[:][:-1]      
    return dem

if __name__ == "__main__":
    rows = 200
    cols = 200
    scale = 30
    # octaves = 2
    # persistence = 0.5
    # lacunarity = 0
    # seed = 6
    octaves = 2
    persistence = 1
    lacunarity = 2.0
    seed = 4
    smoothness = 5
    new_min = -5
    new_max = 4 # Adjust the new maximum height