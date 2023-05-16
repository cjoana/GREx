import h5py 
import numpy as np
import time 
from functools import wraps
import multiprocessing



# Create timeit decorator
def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper


def task(i):
    arr[i] = i +  2**i if i < 10 else i*10 
    print(f"\r {i}: {arr[i]}", end="", flush=True)
    # t = open("pybuff.txt", "a")
    # t.write(f"{i}: {arr[i]}\n")
    # t.close()
    time.sleep(0.01)
    return i, arr[i]



@timeit
def parallel_map(rnge):
    r = pool.map(task, rnge)
    for i, ar in r:
        arr[i] = ar
    print("\nDone, p-map") 


@timeit
def regular_loop(rnge):
    for x in rnge:
        task(x)
    print("\nDone")





if __name__ == '__main__':

    N = 3000
    npools = 100
    arr = np.zeros(N)
    pool = multiprocessing.Pool(npools)


    parallel_map(range(N) )
    print(np.array(arr[:]))


    time.sleep(1)

    arr[:]= arr[:]*0

    regular_loop(range(N) )
    print(np.array(arr[:]))



