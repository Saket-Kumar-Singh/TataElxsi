import sys
sys.path.append("D:\\TataElxsi\\Code\\server")
import process_dem
from .main import *
import make_dem
from .map import *
# import resources
from  .vehicle import *

__all__ = ["vehicle", "process_dem", "p", "make_dem", "map"]

if __name__ == "__main__":
    v = vehicle.vehicle(3)