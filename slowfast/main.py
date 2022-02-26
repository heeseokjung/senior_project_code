import os
import json
import cv2
import numpy as np
import torch
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo
) 
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

num_frames = 32
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
side_size = 256
crop_size = 256
slowfast_alpha = 4

class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors. 
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // slowfast_alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list

transform =  ApplyTransformToKey(
    key="video",
    transform=Compose(
        [
            UniformTemporalSubsample(num_frames),
            Lambda(lambda x: x/255.0),
            NormalizeVideo(mean, std),
            ShortSideScale(
                size=side_size
            ),
            CenterCropVideo(crop_size),
            PackPathway()
        ]
    ),
)


def main():
    model_name = 'slowfast_r50'
    model = torch.hub.load("facebookresearch/pytorchvideo", model=model_name, pretrained=True)
    
    model = model.cuda()
    torch.backends.cudnn.benchmark = False
    model.eval()

    with open('kinetics_classnames.json', 'r') as f:
        kinetics_classnames = json.load(f)

    kinetics_id_to_classname = {}
    for k, v in kinetics_classnames.items():
        kinetics_id_to_classname[v] = str(k).replace('"', '')

    data_path = './dataset/tt0049470/'
    video = os.listdir(data_path)
    video.sort()
    video = [video[i:i+3] for i in range(0, len(video), 3)]

    # for visualization
    fig = plt.figure(figsize=(10,6))
    gs = GridSpec(nrows=3, ncols=2, height_ratios=[1,1,1], width_ratios=[1,1])

    naug = 10
    for i, shot in enumerate(video):
        snippet = []
        for filename in shot:
            for k in range(naug):
                img = cv2.imread(data_path + filename)
                img = img[...,::-1]
                snippet.append(img)

        snippet = torch.Tensor(np.array(snippet))
        snippet = snippet.permute(3, 0, 1, 2) # thwc -> cthw
        video_data = {'video' : snippet}

        video_data = transform(video_data)
        inputs = video_data['video']
        inputs = [i.to('cuda')[None, ...] for i in inputs]

        with torch.no_grad():
            logits = model(inputs).cpu()

        post_act = torch.nn.Softmax(dim=1)
        preds = post_act(logits)
        pred_classes = preds.topk(k=5).indices[0]
        
        label_plot = []
        for i in pred_classes:
            label_plot.append(kinetics_id_to_classname[int(i)])

        # visualization
        plt.suptitle(f'shot {i}')
        plt.subplots_adjust(wspace=2)

        for j, filename in enumerate(shot):
            img = cv2.imread(data_path + filename)
            img = img[...,::-1]
            plt.subplot(gs[j,0]).imshow(img)
            plt.subplot(gs[j,0]).axis('off')
            plt.subplot(gs[j,0]).set_title(f'image {j}')

        plt.subplot(gs[:,1]).barh(label_plot, preds.topk(k=5).values[0])
        plt.subplot(gs[:,1]).set_title('prediction')
        
        plt.draw()
        plt.pause(1)
        plt.clf()

if __name__ == '__main__':
    main()