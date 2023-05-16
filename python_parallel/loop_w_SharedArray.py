import multiprocessing as mp
import numpy as np 
import time


def get_function(x):
    return 1 + x%10

def add_at_index(i):
    """
    https://eli.thegreenplace.net/2012/01/04/shared-counter-with-pythons-multiprocessing
    """
    # if USE_LOCK:
    #     with lock:
    #         for j in range(i):
    #             array[j] += 1
    # else:
    #     for j in range(i):
    #         array[j] += 1

    array[i] = get_function(array[i])
    arrayB[i] = get_function(arrayB[i])

    print(f"\r {array[i]}   {arrayB[i]}", end="")
    # time.sleep(0.001)


def setup(a, l):
    global array, lock
    array = a
    lock = l

if __name__ == "__main__":

    alen = 100000
    array = mp.Array('d', np.arange(alen))
    arrayB = mp.Array('d', np.zeros(alen))

    USE_LOCK = False # True
    lock = mp.Lock() if USE_LOCK else 0
    nums = range(len(array))
    # pool = mp.Pool(100, initializer=setup, initargs=[array, lock])
    pool = mp.Pool(10)   # ~ optimal is num.cpu (=8) + 2
    start = time.time()
    pool.map(add_at_index, nums)
    end = time.time()

    print(" ")
    print(arrayB[:] )
    print("time : ", end - start)