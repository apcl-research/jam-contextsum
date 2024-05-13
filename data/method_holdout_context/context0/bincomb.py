import os
import glob
import numpy as np
from tqdm import tqdm

#data_dir = os.path.join('data', dataset)

def main(data_dir:str = 'bins' # directory of your bin files
        ):


    bins = list()

    for f in tqdm(glob.glob('bins/val*')):
        val = np.memmap(f, dtype=np.uint16, mode='r')
        bins.append(val)

    comb = np.concatenate(bins)

    out = np.memmap('val.bin', dtype=np.uint16, mode='w+', shape=comb.shape)
    out[:] = comb[:]

    for f in tqdm(glob.glob('bins/train*')):
        train = np.memmap(f, dtype=np.uint16, mode='r')
        bins.append(train)

    comb = np.concatenate(bins)

    out = np.memmap('train.bin', dtype=np.uint16, mode='w+', shape=comb.shape)
    out[:] = comb[:]

    print(f'training tokens: {len(out)}')

    

