import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

from Toy_model import toy_model
from Toy_dataset import toy_dataset
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np


def entropy(preds):
 
    preds = F.softmax(preds, dim=-1)
    logp = torch.log(preds + 1e-5)
    entropy = torch.sum(-preds * logp, dim=-1)

    return entropy

    
def osr_test(logits, mode="entropy"):

    if mode == "entropy":
        pred_ent = entropy(logits)
    
    return pred_ent


def feature_stats(inlier_features):

    stats = []
    for features in inlier_features:
        features = np.squeeze(np.array(features))
        print(features.shape)
        mu = np.mean(features, axis=0)
        var = np.cov(features, rowvar=False)

        stats.append((mu, var))
    
    return stats


def knn(inlier_features, outlier_features, inlier_labels, k, mode = "cosine"):

    inlier_features = np.concatenate(inlier_features, axis=0)
    inlier_features = np.squeeze(inlier_features)
    closest = []
    for features in outlier_features:
        if mode == "cosine":
            distances = np.matmul(features, inlier_features.T)
            distances = distances / np.linalg.norm(features, axis=1) / np.linalg.norm(inlier_features, axis=1)
            distances = np.abs(distances)
            distances = np.squeeze(distances)
        else:
            distances = inlier_features - features
            distances = np.linalg.norm(distances, axis=1)
        #idx = np.argpartition(distances, k)                # closest samples
        #idx = idx[:k]  
        idx = (-distances).argsort()[:k]                               
        labels = [inlier_labels[i] for i in idx]           # closest classes
        majority = np.argmax(np.bincount(labels))          # the majority
        #print(idx, labels, majority)
        closest.append(majority)

    return closest


def compare_hist(n1, n2, weights=False):

    n = np.sum(n1)
    normalized_n1 = n1*1.0 / n
    normalized_n2 = n2*1.0 / n

    distance = normalized_n1 - normalized_n2
    if weights == True:
        bins = len(n1)
        weights = np.arange(bins, 0, -1)
        weights = np.divide(weights, n *1.0)
        distance = np.multiply(distance, weights)

    distance = np.sum(np.abs(distance))
    return distance


def distances(stats, inlier_features, outlier_features, mode="mahalanobis"):

    dis_outliers = []
    closest_outliers = []
    dis_all = []
    for features in outlier_features:
        dis_min = 1e10
        idx = 0
        dis_ind = []
        for i, (mu, var) in enumerate(stats):
            #mu, var = stats[0]                             ##### delete
            if mode == "mahalanobis":
                features_normalized = features - mu
                dis =  np.matmul(features_normalized, np.linalg.inv(var))
                dis = np.matmul(dis, np.swapaxes(features_normalized, 0, 1))
                dis = dis[0][0]
            else:
                features = np.squeeze(np.array(features))
                dis = features - mu
                dis = np.sum(np.abs(dis))

            if dis_min > dis:
                dis_min = dis
                idx = i
            dis_ind.append(dis)
        dis_outliers.append(dis_min)
        closest_outliers.append(idx)
        dis_all.append(dis_ind)

    dis_inliers = np.empty([0])
    for c, features in enumerate(inlier_features):
        features = np.squeeze(np.array(features))
        mu, var = stats[c]                               # c
        centers = np.tile(mu, (len(features), 1))
        if mode == "mahalanobis":
            features_normalized = features - centers
            dis =  np.matmul(features_normalized, np.linalg.inv(var))
            dis = np.matmul(dis, np.swapaxes(features_normalized, 0, 1))
        else:
            dis = np.sum(np.abs(features - centers), axis=1)
        dis = np.diag(dis)
        dis_inliers =  np.concatenate((dis_inliers, dis), axis=0)

    return dis_inliers, dis_outliers, closest_outliers, dis_all                      # dis_all [num_features, num_classes]


def similarities(stats, inlier_features, outlier_features):

    s_outliers = []
    for features in outlier_features:
        features = np.squeeze(np.array(features))
        s_min = 1e10
        for mu, var in stats:
            similarities = np.matmul(features, mu.T)
            similarities = similarities / np.sum(np.abs(features)) / np.sum(np.abs(mu))
            similarities = np.abs(similarities)
            if s_min > similarities:
                s_min = similarities
        s_outliers.append(s_min)

    s_inliers = np.empty([0])
    for c, features in enumerate(inlier_features):
        features = np.squeeze(np.array(features))
        mu, var = stats[c]
        centers = np.tile(mu, (len(features), 1))
        similarities = np.matmul(features, centers.T)[:, 0]
        similarities = similarities / np.linalg.norm(mu) / np.linalg.norm(features, axis=1)
        similarities = np.abs(similarities)
        s_inliers =  np.concatenate((s_inliers, similarities), axis=0)

    return s_inliers, s_outliers


if __name__ == "__main__":


    batch_size = 1
    inlier_data_path = "D://projects//open_cross_entropy//code//toy_data_test_inliers"
    outlier_data_path = "D://projects//open_cross_entropy//code//toy_data_test_outliers"
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.RandomHorizontalFlip(),
                                         ])

    label_mapping = {"circle": 0, "rectangle": 1}                   # , "circleRed": 2  , "circleGreen": 2
    num_classes = len(label_mapping)
    dataset_inliers = toy_dataset(inlier_data_path, label_mapping, data_transform)
    #print(len(dataset_inliers))

    outlier_mapping = {"rectangleBlue": 100}
    dataset_outliers = toy_dataset(outlier_data_path, outlier_mapping, data_transform)

    dataset = ConcatDataset([dataset_inliers, dataset_outliers])
    inlier_loader = DataLoader(dataset_inliers, batch_size, num_workers=4, shuffle=True)
    outlier_loader = DataLoader(dataset_outliers, batch_size, num_workers=4, shuffle=True)

    model_path = "D://projects//open_cross_entropy//save1//toy_model3_E1_140"
    model = toy_model(num_classes)
    model.load_state_dict(torch.load(model_path))

    inlier_preds = []
    inlier_preds_osr = []
    inlier_labels = []
    inlier_features = [[] for i in range(num_classes)]
    for i, (image, label) in enumerate(inlier_loader):
        image = image.float()
        label = label.numpy().item()
        image = image.permute(0, 3, 1, 2)
        pred = model(image)
        inlier_features[label].append(pred.detach().numpy())
        pred_outlier = osr_test(pred)
        pred = torch.argmax(pred)

        inlier_labels.append(label)
        inlier_preds.append(pred.detach().numpy().item())
        inlier_preds_osr.append(pred_outlier.detach().numpy().item())

    
    outlier_preds = []
    outlier_preds_osr = []
    outlier_labels = []
    outlier_features = []
    for i, (image, label) in enumerate(outlier_loader):
        image = image.float()
        image = image.permute(0, 3, 1, 2)
        pred = model(image)
        outlier_features.append(pred.detach().numpy())
        pred_outlier = osr_test(pred)
        pred = torch.argmax(pred)

        outlier_labels.append(label.detach().numpy().item())
        outlier_preds.append(pred.detach().numpy().item())
        outlier_preds_osr.append(pred_outlier.detach().numpy().item())


    stats = feature_stats(inlier_features)
    #print(stats)

    d_inliers, d_outliers, closest_md, d_all = distances(stats, inlier_features, outlier_features)
    #d1, d2, d3 = d_inliers[:100], d_inliers[100:200], d_inliers[200]
    #print(np.mean(d1), np.mean(d2), np.mean(d3), np.mean(d_outliers))

    
    d_all = np.array(d_all)
    d_all_mean = np.mean(d_all, axis=0)
    print("d_all_mean", d_all_mean)
    closest_knn = knn(inlier_features, outlier_features, inlier_labels, k=7,  mode="cosine")
    

    n_inlier, _, _ = plt.hist(d_inliers, bins=30, range=(0, 30), alpha=0.5, label="Inlier Similarities")
    n_outlier, _, _ = plt.hist(d_outliers, bins=30, range=(0, 30), alpha=0.5, label="Outlier Similarities")
    plt.xlabel('Distances')
    plt.ylabel('Counts')
    plt.legend(prop ={'size': 10})
    plt.title('Histogram of Mahalanobis Distances', fontsize=15)
    print("Histogram Distance", compare_hist(n_inlier, n_outlier))
    print("Histogram Distance (weights)", compare_hist(n_inlier, n_outlier, weights=True))
    plt.savefig("D://projects//open_cross_entropy//code//hist_distance_E1_140.png")

    plt.close()

    label, counts = np.unique(outlier_preds, return_counts=True)
    plt.bar(x=label, height=counts)
    plt.xlabel('Class Index')
    plt.ylabel('Counts')
    plt.title('Histogram of Preds of Outliers', fontsize=15)
    plt.savefig("D://projects//open_cross_entropy//code//hist_outlier_preds_E1_140.png")

    plt.close()

    label, counts = np.unique(closest_knn, return_counts=True)
    plt.bar(x=label, height=counts)
    plt.xlabel('Class Index')
    plt.ylabel('Counts')
    plt.title('Histogram of Closest of Outliers', fontsize=15)
    plt.savefig("D://projects//open_cross_entropy//code//hist_outlier_closest_knn_E1_140.png")
    
    plt.close()

    label, counts = np.unique(closest_md, return_counts=True)
    plt.bar(x=label, height=counts)
    plt.xlabel('Class Index')
    plt.ylabel('Counts')
    plt.title('Histogram of Predictions on Outliers', fontsize=15)
    plt.savefig("D://projects//open_cross_entropy//code//hist_outlier_closest_md_E1_140.png")
    

    """
    # plot results
    #bins = np.linspace(-0.1, 0.5, 100)
    plt.hist(inlier_preds_osr, bins=100, range=(-1.1e-5, 0), alpha=0.5, label="Inlier Entropies")
    plt.hist(outlier_preds_osr, bins=100, range=(-1.1e-5, 0), alpha=0.5, label="Outlier Entropies")
    plt.legend(prop ={'size': 10})
    plt.xlabel('Entropy')
    plt.ylabel('Counts')
    #plt.title(r'\fontsize{30pt}{3em}\selectfont{}{Histogram of Entropies\r}{\fontsize{18pt}{3em}\selectfont{}(Blue Circle vs Red Rectangles)}')
    plt.title('Histogram of Entropies', fontsize=15)
    #plt.suptitle('(Blue Circle vs Red Rectangles)\n', fontsize=18)
    plt.savefig("D://projects//open_cross_entropy//code//hist_3class.png")
    """
    
        