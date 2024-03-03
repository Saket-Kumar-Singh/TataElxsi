import sim
import math

def connect_to_coppelia():
    # Connect to CoppeliaSim
    clientID = sim.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
    if clientID != -1:
        print('Connected to CoppeliaSim!')
        return clientID
    else:
        print('Failed to connect to CoppeliaSim!')
        return None

def get_terrain_height(clientID, x, y):
    # Get the height of the terrain at position (x, y)
    res, terrainHandle = sim.simxGetObjectHandle(clientID, 'terrain', sim.simx_opmode_blocking)
    res, terrainHeight = sim.simxGetHeightfieldHeight(clientID, terrainHandle, [x, y])
    if res == sim.simx_return_ok:
        return terrainHeight[2]  # Z-coordinate represents height
    else:
        return None

def move_ball(clientID, ballHandle, terrainSize, maxHeight):
    # Move the ball from (0, 0) to the size of the terrain
    stepSize = 0.1
    for x in range(0, terrainSize[0], stepSize):
        for y in range(0, terrainSize[1], stepSize):
            # Get the height of the terrain at position (x, y)
            terrainHeight = get_terrain_height(clientID, x, y)
            if terrainHeight is not None:
                # Set the position of the ball to (x, y, terrainHeight)
                res = sim.simxSetObjectPosition(clientID, ballHandle, -1, [x, y, terrainHeight], sim.simx_opmode_oneshot)
                if res != sim.simx_return_ok:
                    print('Failed to move the ball!')
                    return
                # Check for collision with the terrain
                ballPosition = [x, y, terrainHeight]
                res, collision = sim.simxCheckCollision(clientID, ballHandle, -1, None, None, None, sim.simx_opmode_blocking)
                if res == sim.simx_return_ok and collision:
                    print('Collision detected at position:', ballPosition)
                    return
    print('No collision detected.')

def main():
    clientID = connect_to_coppelia()
    if clientID is None:
        return

    # Get the handle of the ball object
    res, ballHandle = sim.simxGetObjectHandle(clientID, 'ball', sim.simx_opmode_blocking)
    if res != sim.simx_return_ok:
        print('Failed to get the handle of the ball!')
        return

    # Get the size of the terrain
    res, terrainHandle = sim.simxGetObjectHandle(clientID, 'terrain', sim.simx_opmode_blocking)
    res, terrainSize = sim.simxGetObjectFloatParameter(clientID, terrainHandle, sim.sim_objfloatparam_modelbbox_max_z, sim.simx_opmode_blocking)
    if res != sim.simx_return_ok:
        print('Failed to get the size of the terrain!')
        return

    # Get the maximum height of the terrain
    maxHeight = terrainSize[2]

    # Move the ball and check for collisions
    move_ball(clientID, ballHandle, terrainSize, maxHeight)

    # Disconnect from CoppeliaSim
    sim.simxFinish(clientID)

if __name__ == '__main__':
    main()
