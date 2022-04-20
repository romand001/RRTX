import pickle as pkl
import numpy as np
import ast

filename = 'experiment_data_20220419-000607.pkl' # copy filename from your files

data = pkl.load(open(filename, 'rb'))

# read intermitent rrtx data
with open('rrtx_intermitent_data.txt', 'r') as f:
    rrtx_intermitent_data = f.readlines()
    # format is 'time, [r1_path_length, r2_path_length, r3_path_length, r4_path_length]'
    rrtx_intermitent_data = [ast.literal_eval(line) for line in rrtx_intermitent_data]

    print('RRTX Results:')

    times = np.array([line[0] for line in rrtx_intermitent_data])
    print(f'\tAverage time: {np.mean(times):.2f}s')
    print(f'\tMedian time: {np.median(times):.2f}s')
    print(f'\tStandard deviation of time: {np.std(times):.2f}s')

    path_lengths = np.array([line[1] for line in rrtx_intermitent_data])
    print(f'\tAverage path length: {np.mean(path_lengths):.2f}')
    print(f'\tMedian path length: {np.median(path_lengths):.2f}')
    print(f'\tStandard deviation of path length: {np.std(path_lengths):.2f}')
    print(f'\tAverage standard deviation of path length across sims: {np.mean(np.std(path_lengths, axis=0)):.2f}')



# calculate statistics
for algo_name in data.keys():
    if algo_name == 'RRTX' or algo_name == 'Velocity Obstacles':
        continue
    print(f'\n{algo_name} Results:')

    not_none = np.array([i for i in range(len(data[algo_name]['time'])) if data[algo_name]['time'][i] is not None])

    times = np.array(data[algo_name]['time'])[not_none]
    print(f'\tAverage time: {np.mean(times):.2f}s')
    print(f'\tMedian time: {np.median(times):.2f}s')
    print(f'\tStandard deviation of time: {np.std(times):.2f}s')

    path_lengths = np.array([pl for pl in data[algo_name]['path_lengths'] if len(pl) > 0])
    print(f'\tAverage path length: {np.mean(path_lengths):.2f}')
    print(f'\tMedian path length: {np.median(path_lengths):.2f}')
    print(f'\tStandard deviation of path length: {np.std(path_lengths):.2f}')
    print(f'\tAverage standard deviation of path length across sims: {np.mean(np.std(path_lengths, axis=0)):.2f}')

print('Velocity Obstacles Results:')

times = np.array(data['Velocity Obstacles']['time'])
print(f'\tTime: {times[0]}s')

path_lengths = np.array(data['Velocity Obstacles']['path_lengths'])
print(f'\tPath length: {path_lengths[0]:.2f}')

