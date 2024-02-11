from server import models 
import pickle
import time

def make_vehicle(ID):
    k = models.vehicle(ID)
    file = "D:\\TATAELXSI\\app\\server\\myvehicles.pkl"
    f = open(file, "rb")
    vehicle_set = pickle.load(f)
    f.close()
    f = open(file, "wb")
    vehicle_set[ID] = k
    pickle.dump(vehicle_set, f)
    f.close()
    return k

def check_present(k):
    file = "D:\\TATAELXSI\\app\\server\\myvehicles.pkl"
    # file = "..\\server\\myvehicles.pkl"
    f = open(file, "rb")
    vehicle_set = pickle.load(f)
    print(vehicle_set)
    if k in vehicle_set:
        return vehicle_set[k]
    
    else:
        print("Absent")
        return None
    

if __name__ == "__main__":
    # try:
    #     vehicle_1 = make_vehicle(15)
    # except FileNotFoundError:
    #     print("Correction needed")
    vehi1 = make_vehicle(4)
    for i in range(1):

        vehicle_1 = check_present(4)
        vehicle_1.ID = 3

    print(vehicle_1.ID)
    print(vehi1.ID)