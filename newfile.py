import numpy as np
from datetime import datetime 
import time
import pandas as pd
import matplotlib.pyplot as plt

def get_velocity(filename2):
    df = pd.read_csv(filename2, header=None)
    vx_list = []
    vy_list = []
    vz_list = []
    for i in range(11):
        vx = df.iloc[i][0]
        vy = df.iloc[i][2]
        vz = df.iloc[i][4]
        vx_list.append(vx)
        vy_list.append(vy)
        vz_list.append(vz)
    velocity = [vx_list, vy_list, vz_list]
    velocity_array = np.array(velocity)
    return velocity_array


def get_acc(filename2):
    df = pd.read_csv(filename2, header=None)
    ax_list = []
    ay_list = []
    az_list = []
    for i in range(11):
        ax = df.iloc[i][1]
        ay = df.iloc[i][3]
        az = df.iloc[i][5]
        ax_list.append(ax)
        ay_list.append(ay)
        az_list.append(az)
    acc = [ax_list, ay_list, az_list]
    acc_array = np.array(acc)
    return acc_array

def get_position(filename1):
    df = pd.read_csv(filename1, header=None)
    position_list = []
    for i in range(11):
        position = df.iloc[i][:3]
        position_list.append(position)
    pos_array = np.array(position_list)
    return pos_array

def get_location(x, P, las_reached):
    curr_time = datetime.now()
    timedelta =  curr_time - las_reached
    t = timedelta.total_seconds() 
    print("generating location through kalman filter")
    lst = [[0.0 for i in range(9)] for j in range(9)]
    for i in range(9):
        lst[i][i] = 1
    for i in range(3):
        lst[3+i][i] = t
        lst[6+i][i] = 0.5*t*t
        lst[6+i][3+i] = t        
    F = np.array(lst)
    Q = np.eye(9) 
    H = np.eye(9)   
    R = np.eye(9)
    array1 = get_velocity('C:\\Users\\Jahnavi\\combined_file_imu.csv')
    array2 = get_acc('C:\\Users\\Jahnavi\\combined_file_imu.csv')
    imu =  np.concatenate((array1, array2)) # Assuming you have a function to get IMU data
    yk = get_position('C:\\Users\\Jahnavi\\combined_csv_file_gps.csv')
    yk = yk + imu  # Assuming IMU data is added to position
    xk_ = F @ x
    Pk_ = F @ P @ F.T + Q
    Kk = Pk_ @ H.T @ np.linalg.inv((H @ Pk_ @ H.T + R))
    xk = xk_ + Kk @ (yk - H @ xk_)
    Pk = (np.eye(9) - Kk @ H) @ Pk_
    curr_time = datetime.now()
    return xk, Pk, curr_time

def main():
    filename1 = 'C:\\Users\\Jahnavi\\combined_csv_file_gps.csv'
    filename2 = 'C:\\Users\\Jahnavi\\combined_file_imu.csv'
    x = np.zeros((9, 1))
    P = np.eye(9)
    t = datetime.now()
    for i in range(11):
        x, P, t = get_location(x, P, t)
        print(x[0])
        print(x[1])
        print(x[2])
        print("---------------------------------------")  
        time.sleep(1)
           
main()