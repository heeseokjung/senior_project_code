import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))

def avg_pooling(encoded_shots):
    return np.sum(encoded_shots, axis=0) / len(encoded_shots)

def DTW(encoded_shots, center, K):
    dtw = []
    bs = np.arange(-K+1, K, dtype=int)
    l_end = encoded_shots[max(0, center-K)]
    r_end = encoded_shots[min(len(encoded_shots)-1, center+K)]
    for b in bs:
        lsum = 0.
        lrange = center + np.arange(-K+1, b+1, dtype=int)
        lrange = np.clip(lrange, 0, len(encoded_shots)-1)

        rsum = 0.
        rrange = center + np.arange(b+1, K, dtype=int)
        rrange = np.clip(rrange, 0, len(encoded_shots)-1)
        
        for i in lrange:
            lsum += sim(l_end, encoded_shots[i])
        lsum /= (b + K)

        for i in rrange:
            rsum += sim(r_end, encoded_shots[i])
        if K - b - 1 != 0:
            rsum /= (K - b - 1)

        dtw.append(lsum + rsum)

    dtw = np.array(dtw)
    idx = np.argmax(dtw)

    return bs[idx]

def avg_precision(pr):
    ap = 0.
    pr = sorted(pr, key=lambda x: x[1])
    for i in range(1, len(pr)):
        ap += (pr[i][1] - pr[i-1][1])*pr[i][0]
    ap += pr[0][1]*pr[0][0]

    return ap

''' fixed length ver. '''
'''
def scene_boundary_classifier(encoded_shots, threshold):
    K = 12
    boundaries = []
    for center in range(0, len(encoded_shots)-1):
        idx = center + np.arange(-K, K+1, dtype=int)
        idx = np.clip(idx, 0, len(encoded_shots)-1)
        s_left = avg_pooling(encoded_shots[idx[:K+1]])
        s_right = avg_pooling(encoded_shots[idx[K+1:]])
        
        similarity = sim(s_left, s_right)
        if similarity <= threshold:
            boundaries.append(center)

    return boundaries
'''

def scene_boundary_classifier(encoded_shots, threshold):
    K = 2
    boundaries = []
    for center in range(0, len(encoded_shots)-1):
        idx = center + np.arange(-K, K+1, dtype=int)
        idx = np.clip(idx, 0, len(encoded_shots)-1)
        b = DTW(encoded_shots, center, K)
        s_left = avg_pooling(encoded_shots[idx[:K+b+1]])
        s_right = avg_pooling(encoded_shots[idx[K+b+1:]])
        
        similarity = sim(s_left, s_right)
        if similarity <= threshold:
            boundaries.append(center)

    return boundaries

def evaluate(boundaries, annotation):
    sseg = set()
    for scene in annotation.values():
        shots = scene['shot']
        sseg.add(int(shots[0])-1)
        sseg.add(int(shots[len(shots)-1]))

    count = 0
    for boundary in boundaries:
        if boundary in sseg:
            count += 1

    precision = count / len(boundaries) if len(boundaries) != 0 else 0.
    recall = count / len(sseg)

    return precision, recall

def main():
    with open('scene_movie318.json', 'r') as f:
        sseg = json.load(f)
    
    sum_ap = 0
    thresholds = np.arange(-1., 1., .05)
    for movie in tqdm(sseg.keys()):
        path = './resnet_50_1000/' + movie + '.npy'
        encoded_shots = np.load(path)

        pr = []
        for threshold in thresholds:
            boundaries = scene_boundary_classifier(encoded_shots, threshold)
            precision, recall = evaluate(boundaries, sseg[movie])
            pr.append((precision, recall))
        
        sum_ap += avg_precision(pr)
    
    print(f'mAP: {sum_ap / len(sseg)}')

if __name__ == '__main__':
    main()