from pathlib import Path 
import os 
import multiprocessing

dataset_path = Path('/miele/timur/edit_dataset/')


processes = []
files = {}
num_processes = 10  # Number of processes you want to start

all_subsets = next(os.walk(dataset_path))[1]
step = len(all_subsets) // (num_processes - 1)
diction = {}
names = ['_'.join(v.split('_')[1:-1]) for v in all_subsets]
names = set(names)
print(names)
print(len(names))

# def worker_function(index,): 
#     import code; code.interact(local=locals())
#     names = ['_'.join(v.split('_')[1:-1]) for v in all_subsets[index:index+step]]
#     name = set(names)
    
#     diction[index] = name

# # Create and start multiple processes
# for i in range(num_processes):
#     index = i * step
#     process = multiprocessing.Process(target=worker_function, args=(index,))
#     processes.append(process)
#     process.start()

# # Wait for all processes to finish
# for process in processes:
#     process.join()

# print('All processes completed.')

# overall_set = set()
# for k, v in diction.items():
#     overall_set.update(v)

# print(overall_set)
# print(len(overall_set))
# import code; code.interact(local=locals())

