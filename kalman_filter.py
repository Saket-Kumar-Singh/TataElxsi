import numpy as np
from datetime import datetime 
import time
import pandas as pd
import matplotlib.pyplot as plt
from time import sleep

def get_velocity(filename2):
    df = pd.read_csv(filename2, header=None)
    vx_list = []
    vy_list = []
    vz_list = []
    for i in range(0, 70, 7):
        vx_list.append(*df[i+0].values)
        vy_list.append(*df[i+2].values)
        vz_list.append(*df[i+4].values)
    # print(vx_list)
    # sleep(2)    
    velocity = [vx_list, vy_list, vz_list]
    velocity_array = np.array(velocity)
    # print(velocity_array.shape)
    return velocity_array


def get_acc(filename2):
    df = pd.read_csv(filename2, header=None)
    
    ax_list = []
    ay_list = []
    az_list = []
    for i in range(0, 70, 7):
        ax_list.append(*df[i+1].values)
        ay_list.append(*df[i+3].values)
        az_list.append(*df[i+5].values)
    # print(vx_list)
    # sleep(2)   
    acc_array = [ax_list, ay_list, az_list]
    # velocity = [vx_list, vy_list, vz_list]
    # velocity_array = np.array(velocity)
    # print(velocity_array.shape)
    return np.array(acc_array)

def get_position(filename1):
    df = pd.read_csv(filename1, header=None)
    arr = []
    temp = []
    # print(df[0].value)
    x_init = (*df[0].values, *df[1].values, *df[2].values)
    for i in range(33):
        if(i%3 == 0):
            print(*df[i].values)
        temp.append(*df[i].values - x_init[i%3])
        p = temp[-1]
        if not i%3 == 2: 
            p = 111000*p
        else :  
            p = 1000*p
        temp.pop(-1)
        temp.append(p)
        if not (i+1)%3:
            arr.append(temp)
            temp = []
    print(np.array(arr).shape)        
    # sleep(2)
    position_list = []
    for i in range(11):
        position = arr[i][:3]
        position_list.append(position)
    pos_array = np.array(position_list)
    # sleep(10)
    return pos_array

def kalman(x, x_las, P,  array1, array2):
    curr_time = datetime.now()
    t = 0.1
    # print("generating location through kalman filter")
    lst = [[0.0 for i in range(3)] for j in range(3)]
    for i in range(3):
        lst[i][i] = 1
    F = np.array(lst)
    lst = [[0.0 for i in range(6)] for j in range(3)]
    for i in range(3):
        lst[i][i] = t
        lst[i][+i] = 0.5*t*t
        # lst[3+i][3+i] = t        
    G = np.array(lst)
    Q = np.eye(3) 
    H = np.eye(3)   
    R = np.eye(3)

    # Q = 0.1*Q
    # H = 
    # array1 = get_velocity('D:\\TataElxsi\\combined_file_imu.csv')
    # array2 = get_acc('D:\\TataElxsi\\combined_file_imu.csv')
    # imu =  np.concatenate((array1, array2)) # Assuming you have a function to get IMU data
    # yk = get_position('D:\\TataElxsi\\combined_csv_file_gps.csv')
    imu = np.concatenate((array1, array2))
    # yk =   np.concatenate((np.array(x_las) , imu)) # Assuming IMU data is added to position
    yk = np.array(x_las)

    xk_ = F @ yk + G @ imu
    Pk_ = F @ P @ F.T + Q

    Kk = Pk_ @ H.T @ np.linalg.inv((H @ Pk_ @ H.T + R))
    xk = xk_ + Kk @ (x - H @ xk_)
    Pk = (np.eye(3) - Kk @ H) @ Pk_
    curr_time = datetime.now()
    return xk, Pk, curr_time

def main():
    filename1 = 'D:\\TataElxsi\\combined_csv_file_gps.csv'
    filename2 = 'D:\\TataElxsi\\combined_file_imu.csv'
    arr_pos = get_position(filename1)
    arr1 = get_velocity(filename2)
    arr2 = get_acc(filename2)
    # print(arr_pos.sha/pe)
    x_pos = arr_pos[:, 0]
    y_pos = arr_pos[:, 1]
    z_pos = arr_pos[:, 2]
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
    for i in range(1,11):
        x = arr_pos[i]
        ar1 = arr1[:, i-1]
        ar2 = arr2[:, i-1]
        print(x[2])
        x, P, curr_time = kalman(x,x_las, P, ar1, ar2)
        x_las = x[:3]
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