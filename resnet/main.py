from cProfile import label
import os
import re
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import models, transforms
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def main():
    model = models.resnet50(pretrained=True)
    model = model.cuda()
    torch.backends.cudnn.benchmark = False
    model.eval()

    data_path = './dataset/tt0049470/'
    data_list = os.listdir(data_path)
    data_list.sort()
    data_list = [data_list[i:i+3] for i in range(0, len(data_list), 3)]

    with open('imagenet_classes.txt', 'r') as f:
        categories = [s.strip() for s in f.readlines()]

    # for visualization
    fig = plt.figure(figsize=(10,6))
    gs = GridSpec(nrows=3, ncols=2, height_ratios=[1,1,1], width_ratios=[1,1])

    for i, shot in enumerate(data_list):
        filename = shot[1]
        input_image = Image.open(data_path + filename)
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')

        with torch.no_grad():
            output = model(input_batch)
        
        output = output.squeeze()
        probabilities = torch.nn.functional.softmax(output, dim=0)
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        
        label_plot = []
        pred_plot = []
        for i in range(top5_prob.size(0)):
            label_plot.append(categories[top5_catid[i]])
            pred_plot.append(top5_prob[i].item())

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
        plt.pause(2)
        plt.clf()

        

if __name__ == '__main__':
    main()