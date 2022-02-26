import os
import re
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from model import S3D

def transform(snippet):
    ''' stack & noralization '''
    snippet = np.concatenate(snippet, axis=-1)
    snippet = torch.from_numpy(snippet).permute(2, 0, 1).contiguous().float()
    snippet = snippet.mul_(2.).sub_(255).div(255)

    return snippet.view(1,-1,3,snippet.size(1),snippet.size(2)).permute(0,2,1,3,4)

def main():
    model = S3D(400)
    weight_path = './S3D_kinetics400.pt'

    weight_dict = torch.load(weight_path)
    model_dict = model.state_dict()

    for name, param in weight_dict.items():
        if 'module' in name:
            name = '.'.join(name.split('.')[1:])
        if name in model_dict:
            if param.size() == model_dict[name].size():
                model_dict[name].copy_(param)
            else:
                print(' size? ' + name, param.size(), model_dict[name].size())
        else:
            print(' name? ' + name)
    print('loaded')

    labels = []
    with open('./label_map.txt', 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            labels.append(line.strip())

    model = model.cuda()
    torch.backends.cudnn.benchmark = False
    model.eval()

    data_path = './dataset/tt0049470/'
    data_list = os.listdir(data_path)
    data_list.sort()
    data_list = [data_list[i:i+3] for i in range(0, len(data_list), 3)]

    # for visualization
    fig = plt.figure(figsize=(10,6))
    gs = GridSpec(nrows=3, ncols=2, height_ratios=[1,1,1], width_ratios=[1,1])

    naug = 10
    for i, shot in enumerate(data_list):
        snippet = []
        for filename in shot:
            for k in range(naug):
                img = cv2.imread(data_path + filename)
                img = img[...,::-1]
                snippet.append(img)

        clip = transform(snippet)

        with torch.no_grad():
            logits = model(clip.cuda()).cpu().data[0]

        prediction = torch.softmax(logits, 0).numpy()
        sorted_indices = np.argsort(prediction)[::-1][:5]
        
        label_plot = []
        pred_plot = []
        for idx in sorted_indices:
            label_plot.append(labels[idx])
            pred_plot.append(prediction[idx])

        # visualization
        plt.suptitle(f'shot {i}')
        plt.subplots_adjust(wspace=2)

        for j, filename in enumerate(shot):
            img = cv2.imread(data_path + filename)
            img = img[...,::-1]
            plt.subplot(gs[j,0]).imshow(img)
            plt.subplot(gs[j,0]).axis('off')
            plt.subplot(gs[j,0]).set_title(f'image {j}')

        plt.subplot(gs[:,1]).barh(label_plot, pred_plot)
        plt.subplot(gs[:,1]).set_title('prediction')
        
        plt.draw()
        plt.pause(1)
        plt.clf()

if __name__ == '__main__':
    main()