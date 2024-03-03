import numpy as np
from datetime import datetime 
import time
import pandas as pd
import matplotlib.pyplot as plt
from time import sleep

def get_velocity(filename2):
    df = pd.read_csv(filename2, header=None)
    df = np.array(df)
    vx_list = []
    vy_list = []
    vz_list = []
    for i in range(df.shape[0]):
        vx_list.append(df[i][0])
        vy_list.append(df[i][2])
        vz_list.append(df[i][4])
    velocity = [vx_list, vy_list, vz_list]
    velocity_array = np.array(velocity)
    print(velocity_array)
    print("-----------------")
    # print(velocity_array.shape)
    return velocity_array


def get_acc(filename2):
    df = pd.read_csv(filename2, header=None)
    df = np.array(df)
    ax_list = []
    ay_list = []
    az_list = []
    for i in range(df.shape[0]):
        ax_list.append(df[i][1])
        ay_list.append(df[i][3])
        az_list.append(df[i][5])
    # sleep(2)   
    acc_array = [ax_list, ay_list, az_list]
    print(acc_array)    
    # print(vx_list)
    # velocity = [vx_list, vy_list, vz_list]
    # velocity_array = np.array(velocity)
    # print(velocity_array.shape)
    return np.array(acc_array)

def get_position(filename1):
    df = pd.read_csv(filename1, header=None)
    arr = []
    temp = []
    # print(df[0].value)
    df = np.array(df)
    x_init = (df[0][0], df[0][1], df[0][2])
    arr = []
    # print(x_init)
    for i in range(df.shape[0]):
        p = df[i] - np.array(x_init)
        p[0] = 111000*p[0]
        p[1] = 100000*p[1]
        p[2] = 1000*p[2]
        arr.append(p)
    arr = np.array(arr)
    print(np.array(arr).shape)        

    print(arr)
    print("--------------------")
    return arr

def kalman(x, x_las, P,  array1, array2, t = 0.1):
    curr_time = datetime.now()
    # print("generating location through kalman filter")
    lst = [[0.0 for i in range(3)] for j in range(3)]
    for i in range(3):
        lst[i][i] = 1
    F = np.array(lst)
    lst = [[0.0 for i in range(6)] for j in range(3)]
    for i in range(3):
        lst[i][i] = t
        lst[i][3+i] = 0.5*t*t
        # lst[3+i][3+i] = t        
    G = np.array(lst)
    Q = np.eye(3) 
    H = np.eye(3)   
    R = np.eye(3)
    imu = np.concatenate((array1, array2))
    yk = np.array(x_las)

    xk_ = F @ yk + G @ imu
    Pk_ = F @ P @ F.T + Q

    Kk = Pk_ @ H.T @ np.linalg.inv((H @ Pk_ @ H.T + R))
    xk = xk_ + Kk @ (x - H @ xk_)
    Pk = (np.eye(3) - Kk @ H) @ Pk_
    curr_time = datetime.now()
    return xk, Pk, curr_time

def main():
    filename1 = 'D:\\TataElxsi\\final_gps.csv'
    filename2 = 'D:\\TataElxsi\\final_imu.csv'
    arr_pos = get_position(filename1)
    arr1 = get_velocity(filename2)
    arr2 = get_acc(filename2)
    num_data = 30
    # print(arr_pos.sha/pe)
    x_pos = arr_pos[:num_data, 0]
    y_pos = arr_pos[:num_data, 1]
    z_pos = arr_pos[:num_data, 2]
    # print(np.array(x_pos).shape)
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ax.scatter(x_pos, y_pos, z_pos)
    ax.plot(x_pos, y_pos, z_pos)
    # x =   np.zeros((9, 1))
    P = np.zeros((3,3))
    t = datetime.now()
    x_pred = []
    y_pred = []
    z_pred = []
    x_las = arr_pos[0]
    x_pred.append(x_las[0])
    y_pred.append(x_las[1])
    z_pred.append(x_las[2])
    # x_pred.append(x_las)
    for i in range(1, num_data):
        x = arr_pos[i]
        ar1 = arr1[:, i-1]
        ar2 = arr2[:, i-1]
        print(x[2])
        x, P, curr_time = kalman(x,x_las, P, ar1, ar2)
        x_las = x
        print(ar1[2])
        print(ar2[2])
        # print(x[0])
        # print(x[1])
        x_pred.append(x[0])
        y_pred.append(x[1])
        z_pred.append(x[2])
        # x_pred.append(x_las)
        print("---------------------------------------")  
        time.sleep(0.1)

    print(z_pred)
    ax.scatter3D(x_pred, y_pred, z_pred)       
    ax.plot(x_pred, y_pred, z_pred)       
    plt.show()

if __name__ == "__main__":     
    main()