#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 13:57:45 2020

@author: zhi
"""


import numpy as np
import random
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle 

import torch
import torch.nn as nn



def pca(inMat, nComponents):
    
    # It is better to make PCA transformation before tSNE
    pcaFunction = PCA(nComponents)
    outMat = pcaFunction.fit_transform(inMat)

    return outMat    
    
    

def tSNE(inMat, nComponents):
    """
    The function used to visualize the high-dimensional hyper points 
    with t-SNE (t-distributed stochastic neighbor embedding)
    https://towardsdatascience.com/why-you-are-using-t-sne-wrong-502412aab0c0
    https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b
    """
    
    inEmbedded = TSNE(n_components=nComponents, perplexity=10).fit_transform(inMat)
    return inEmbedded
    
    
    
if __name__ == "__main__":
    
    
    num_classes = 3
    featurePath = "D://projects//open_cross_entropy//save//toy_model_E2_999"
    feature_to_visulize = "linear1"
    
    with open(featurePath, "rb") as f:
        featuresTest, labelsTest = pickle.load(f)
 
    print(len(featuresTest))
    featuresTest = [featureTest[feature_to_visulize].detach().numpy() for featureTest in featuresTest]
    featuresTest = np.squeeze(np.array(featuresTest))
    print(featuresTest.shape)
    if "conv" in feature_to_visulize:
       featuresTest = np.reshape(featuresTest, (featuresTest.shape[0], -1))

        
    allLabels = []
    
    featuresSNE = np.squeeze(np.array(featuresTest))                                     
    #print(featuresSNE.shape)
    #featuresSNE = pca(featuresTest, 10)       #10
    featuresSNE = tSNE(featuresSNE, 2)
    #featuresSNE = np.concatenate((featuresSNE, features), 0)

    #featuresSNE = np.squeeze(featuresTest)
    
        
    for l in labelsTest:
        if l >= num_classes:
            allLabels.append(1000)
        else:
            allLabels.append(l.item())
    
    allLabels = np.squeeze(np.reshape(np.array(allLabels), [1, -1]))
    print(allLabels)
    
    f = {"feature_1": featuresSNE[:, 0], 
         "feature_2": featuresSNE[:, 1],
         "label": allLabels}
    
    fp = pd.DataFrame(f)
    
    a4_dims = (8, 6)
    fig, ax = plt.subplots(figsize=a4_dims)
    
    sns.scatterplot(ax=ax, x="feature_1", y="feature_2", hue="label",
                    palette=['blue','red','orange', "k"], data=fp, 
                    legend="full", alpha=0.5)
    fig.savefig("D://projects//open_cross_entropy//code//features//toy_model_E2_199_linear1.png")
    
    
    """
    https://medium.com/swlh/how-to-create-a-seaborn-palette-that-highlights-maximum-value-f614aecd706b
    
    'green','orange','brown','blue','red', 'yellow', 'pink', 'purple', 'c', 'grey'
    ,'brown','blue','red', 'yellow', 'pink', 'purple', 'c', 'grey',
                            'rosybrown', 'm', 'y', 'tan', 'lime', 'azure', 'sky', 'darkgreen',
                            'grape', 'jade'
    sns.color_palette("hls", num_classes)
    """