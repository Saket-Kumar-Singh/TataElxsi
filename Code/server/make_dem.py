import os
import numpy as np
import matplotlib.pyplot as plt
import noise
from scipy.ndimage import gaussian_filter

"""This is a file to generate DEM run this if dem is not already present..."""

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



if __name__ == "__main__":
    # Example usage
    rows = 150
    cols = 150
    scale = 30
    # octaves = 2
    # persistence = 0.5
    # lacunarity = 0
    # seed = 6
    octaves = 2
    persistence = 1
    lacunarity = 2.0
    seed = 4
    smoothness = 2
    new_min = -5
    new_max = 4 # Adjust the new maximum height

    smooth_dem = generate_smooth_dem(rows, cols, scale, octaves, persistence, lacunarity, seed, smoothness, new_min, new_max)
    plot_dem(smooth_dem)
    file = open("DEM.txt", 'a')
    for i in range(len(smooth_dem)):
      for j in range(len(smooth_dem[0])):
        file.write(str(smooth_dem[i][j]))
        file.write(" ")
      file.write("\n")
    file.close()

