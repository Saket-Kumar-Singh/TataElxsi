import read_serial
import time

def loca():
    msg = read_serial.loca()
    msg[2] = msg[2][:-2]
    res = {
        "x" : int(msg[0]),
        "y" : int(msg[1]),
        "z" : int(msg[2]),
    }
    return res

if __name__ == "__main__":
    msg = read_serial.loca()
    tme = time.time()
    print("We are currently at : ", msg)
    msg[2] = msg[2][:-2]
    res = {
        "x" : int(msg[0]),
        "y" : int(msg[1]),
        "z" : int(msg[2]),
    }
    tme = time.time() - tme
    print(res)
    print(f"Generated results in {tme} seconds")