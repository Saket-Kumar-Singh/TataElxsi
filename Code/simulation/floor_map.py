import numpy as np
import matplotlib.pyplot as plt

# Function to read DEM data from text file
def read_dem(file_path):
    with open(file_path, 'r') as file:
        dem_data = [list(map(float, line.split())) for line in file]
    return np.array(dem_data)

# Function to convert DEM data to PNG image
def dem_to_png(dem_data, output_file):
    plt.colorbar(label='Elevation')
    plt.savefig(output_file)
    plt.close()
    plt.imshow(dem_data, cmap='terrain')

# Example usage
if __name__ == "__main__":
    # Provide the path to your DEM text file
    dem_file_path = 'D:\\TataElxsi\\server\\DEM.txt'

    # Read DEM data
    dem_data = read_dem(dem_file_path)

    plt.imshow(dem_data, cmap='terrain')  # 'terrain' colormap for elevation-like colors
    plt.colorbar(label='Elevation (meters)')  # Add a color bar indicating elevation values
    plt.title('Digital Elevation Model (DEM)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)  # Add grid lines for better visualization
    plt.savefig('DEM.png')  # Save the plot as a PNG image
    plt.show()
