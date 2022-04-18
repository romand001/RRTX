import pickle as pkl
import numpy as np

filename = 'experiment_data_20220418-133327.pkl' # copy filename from your files

data = pkl.load(open(filename, 'rb'))

# calculate statistics
for algo_name in data.keys():
    print(f'\n{algo_name} Results:')

    times = np.array(data[algo_name]['time'])
    print(f'\tAverage time: {np.mean(times):.2f}s')
    print(f'\tMedian time: {np.median(times):.2f}s')
    print(f'\tStandard deviation of time: {np.std(times):.2f}s')

    path_lengths = np.array(data[algo_name]['path_lengths'])
    print(f'\tAverage path length: {np.mean(path_lengths):.2f}')
    print(f'\tMedian path length: {np.median(path_lengths):.2f}')
    print(f'\tStandard deviation of path length: {np.std(path_lengths):.2f}')
    print(f'\tAverage standard deviation of path length across sims: {np.mean(np.std(path_lengths, axis=0)):.2f}')

