import time
import threading
from data_sending import conn, make_conn
from get_location import loca

pos = []
count = 0 
def loop_process():
    while True:
        print("Looping process is running...")
        time.sleep(1)

def send_data():
    while True:
        print("sent_data")
        global pos
        if len(pos) == 0:
            continue
        conn(pos[0])
        pos.pop(0)
        time.sleep(1)

def get_data():
    while True:    
        print("getting_data")
        global pos
        global count
        # p = loca()
        p = f"this is message {count}"
        count = count+1
        pos.append(p)
        time.sleep(1)


def threading_process():
    print ("making connection")
    make_conn()
    print("enter threading")
    threads = []
    t = threading.Thread(target=send_data)
    threads.append(t)
    t = threading.Thread(target=get_data)
    threads.append(t)
    for t in threads:
        t.start()

    # Wait for all threads to finish
    for t in threads:
        t.join()
