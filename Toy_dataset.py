import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class toy_dataset(Dataset):

    def __init__(self, data_path, label_map, transform=None):

        self.images = []
        self.labels = []
        self.transform = transform
        os.chdir(data_path)

        for f in os.listdir(data_path):

            img = cv2.imread(f)
            label = f.split("_")[0]
            if label not in label_map.keys():
                #print(f)
                continue
            self.images.append(img)
            label = label_map[label]
            self.labels.append(label)

    def __getitem__(self, index):

        image = self.images[index]
        image = self.transform(image)
        #image = (image - 128.) / 128.
        return self.images[index], self.labels[index]

    def __len__(self):

        return len(self.images)




if __name__ == "__main__":
    data_path = "D://projects//open_cross_entropy//code//toy_data"
    label_mapping = {"circle": 0, "rectangle": 1, "circleRed": 2} 
    data = toy_dataset(data_path, label_mapping)

    images = np.array(data.images)
    images = np.reshape(images, (-1, 3))
    print(np.mean(images, axis=0))
    print(np.std(images, axis=0))
    print(np.max(images, axis=0))
    print(np.min(images, axis=0))
