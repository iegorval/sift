import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ ==  '__main__':
    #data = pd.read_csv('test_opencv.txt', sep=",", header=None)
    #data.columns = ['octave', 'layer', 'sigma']
    #data['placeholder'] = 1
    #print(data.groupby(['octave', 'layer']).placeholder.sum())
    scene = 3

    cpu1 = pd.read_csv(f'log_cpu_{scene}.txt', sep=',', header=None)
    cpu1['scene'] = f'scene_{scene}'
    cpu1['version'] = 'cpu_sequential'
    gpu1 = pd.read_csv(f'log_gpu_{scene}.txt', sep=',', header=None)
    gpu1['scene'] = f'scene_{scene}'
    gpu1['version'] = 'gpu_parallel'
    cv1 = pd.read_csv(f'log_cv_{scene}.txt', sep=',', header=None)
    cv1['scene'] = f'scene_{scene}'
    cv1['version'] = 'cpu_parallel'

    cpu1.columns = ['duration', 'keypoints', 'scene', 'version']
    cv1.columns = ['duration', 'keypoints', 'scene', 'version']
    gpu1.columns = ['duration', 'keypoints', 'scene', 'version']

    cpu1['duration'] /= 1000
    cv1['duration'] /= 1000
    gpu1['duration'] /= 1000

    x = np.arange(len(cv1)) + 1

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('#images')
    ax1.set_ylabel('duration parallel [s]')
    lns1 = ax1.plot(x, cv1['duration'].cumsum(), color='orange', alpha=0.7, label='cpu_parallel')
    ax1.scatter(x, cv1['duration'].cumsum(), color='orange')
    lns2 = ax1.plot(x, gpu1['duration'].cumsum(), color='green', alpha=0.7, label='gpu_parallel')
    ax1.scatter(x, gpu1['duration'].cumsum(), color='green')

    ax2 = ax1.twinx() 
    ax2.set_ylabel('duration sequential [s]') 
    lns3 = ax2.plot(x, cpu1['duration'].cumsum(), color='red', alpha=0.7, label='cpu_sequential')
    ax2.scatter(x, cpu1['duration'].cumsum(), color='red')

    lns = lns3 + lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    print(cpu1.duration.mean())
    print(cpu1.keypoints.mean())
    
    print(cv1.duration.mean())
    print(cv1.keypoints.mean())

    print(gpu1.duration.mean())
    print(gpu1.keypoints.mean())
    #plt.show()
