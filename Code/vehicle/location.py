import numpy as np
from datetime import datetime 
import time
import matplotlib.pyplot as plt
# def get_data():
#     return [4, 5, 6, 1, 0, 0]

data_matrix = np.array([
    [0.0, 2.2, 4.4, 6.8, 9.2, 11.8, 14.4, 17.2, 20.0, 23.0],  # X Position
    [0.0, 1.6, 3.2, 4.9, 6.5, 8.2, 10.0, 11.8, 13.6, 15.5],  # Y Position
    [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9],      # Z Position
    [2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8],      # X Velocity
    [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4],      # Y Velocity
    [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],      # Z Velocity
    [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],      # X Acceleration
    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],      # Y Acceleration
    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]       # Z Acceleration
])

data_matrix = data_matrix.T

def get_position(flag):
    flag += 1
    return data_matrix[flag]


def get_location(x,P, las_reached, flag):
    curr_time = datetime.now()
    timedelta =  curr_time - las_reached
    t = timedelta.total_seconds() 
    # t = (las_reached.second * 10**6 + las_reached.microseconds - curr_time.second * 10 ** 6 - curr_time + ) / (10**6)
    print("generating location through kalman filter")
    # result from imu = [x, y, z, vx, vy, vz, az, ay, az] 
    # v = imu[3:6] (1X3)
    # a = imu[6:9] (1X3)
    lst = [[0.0 for i in range(9)] for j in range(9)]
    for i in range(9):
        lst[i][i] = 1
    for i in range(3):
        lst[3+i][i] = t
        lst[6+i][i] = 0.5*t*t
        lst[6+i][3+i] = t        
    # t = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0], [0, t, 0], [0, 0, t]])
    F = np.array(lst)
    l = [1, 2, 3]
    if type(x) != type(np.array(l)):
        x = np.array(x)

    Q = np.eye(9) 
    H = np.eye(9)   
    R = np.eye(9)

    # imu = get_data()
    yk = get_position(flag)
    # yk = yk + imu
    # print(yk)
    
    xk_ = F @ x
    Pk_ = F @ P @ F.T + Q
    Kk = Pk_ @ H.T @ np.linalg.inv((H @ Pk_ @ H.T + R))
    xk = xk_ + Kk @ (yk - H @ xk_)
    Pk = (np.eye(9) - Kk @ H) @ Pk_
    curr_time = datetime.now()
    return xk, Pk, curr_time

    # except :
    #     # print("error")
    #     return 0, 0, datetime.now()
    # yk = imu[:3]


if __name__ == "__main__":
    # x,y = get_location()
    x = [0 for i in range(9)]
    x = np.array(x).reshape(9, 1)
    P = np.eye(9)
    t = datetime.now()
    x_pos = []
    y_pos = []
    t_pos = []
    flag = 0
    for i in range(5):
        x, P, t = get_location(x, P, t, flag)
        flag += 1
        time.sleep(1)
        x_pos.append(x[1][0])
        y_pos.append(x[1][0])
        t_pos.append(t)
    
    fig1 = plt.plot(t_pos, x_pos, label = "X Pos")
    fig2 = plt.plot(t_pos, y_pos, label = "Y Pos")
    plt.show()    
        # print("The value of x is ", x)        