import multiprocessing


import multiprocessing
import time
import os

# Function for the worker processes
def worker_function(index):
    os.system(f'python create_pair.py --start_index {str(index * 5000 + 35000)}')
    

if __name__ == '__main__':
    processes = []
    num_processes = 20  # Number of processes you want to start

    # Create and start multiple processes
    for i in range(num_processes):
        process = multiprocessing.Process(target=worker_function, args=(i,))
        processes.append(process)
        process.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()

    print('All processes completed.')
