import pickle
import matplotlib.pyplot as plt
import numpy as np


def class_center(class_features):

    """
    class_features: [num_samples, feature_dim]
    """
    class_features = np.array(class_features)
    center = np.mean(class_features, axis=0)
    return center


def cosine_distance(features, center):

    """
    features: [num_samples, feature_dim]
    center: [feature_dim]
    """
    features = np.array(features)
    centers = np.tile(center, (len(features), 1))
    similarities = np.matmul(features, centers.T)[:, 0]
    center_norm = np.linalg.norm(center)
    similarities = similarities / center_norm
    features_norm = np.linalg.norm(features, axis=1)
    similarities = np.divide(similarities, features_norm)

    return similarities


def seperate_class(featuresTest, labelsTest, num_classes):

    seperated_features = [[] for i in range(num_classes)]
    for i, l in enumerate(labelsTest):
        #print(featuresTest[i])
        seperated_features[l.item()].append(featuresTest[i])

    return seperated_features


if __name__ == "__main__":

    num_classes = 3
    featurePath = "D://projects//open_cross_entropy//save//toy_model_train3"
    feature_to_visulize = "linear3"
    
    with open(featurePath, "rb") as f:
        featuresTrain, labelsTrain = pickle.load(f)
 
    featuresTrain = [featureTrain[feature_to_visulize].detach().numpy() for featureTrain in featuresTrain]
    featuresTrain = np.squeeze(np.array(featuresTrain))
    print(featuresTrain.shape)

    seperated_features = seperate_class(featuresTrain, labelsTrain, num_classes)
    
    centers = []
    for class_features in seperated_features:
        center = class_center(class_features)
        centers.append(center)
        similarity = cosine_distance(class_features, center)
        print(similarity)


    featurePath_test = "D://projects//open_cross_entropy//save//toy_model_test3"

    with open(featurePath_test, "rb") as f:
        featuresTest, labelsTest = pickle.load(f)

    featuresTest = [featureTest[feature_to_visulize].detach().numpy() for featureTest in featuresTest]
    featuresTest = np.squeeze(np.array(featuresTest))
    seperated_features_test = seperate_class(featuresTest, labelsTest, num_classes+3)
    

    similarity11 = cosine_distance(seperated_features_test[0], centers[0])
    similarity22 = cosine_distance(seperated_features_test[1], centers[1])
    similarity33 = cosine_distance(seperated_features_test[2], centers[2])

    similarity41 = cosine_distance(seperated_features_test[3], centers[0])
    similarity42 = cosine_distance(seperated_features_test[3], centers[1])
    similarity43 = cosine_distance(seperated_features_test[3], centers[2])

    similarity51 = cosine_distance(seperated_features_test[4], centers[0])
    similarity52 = cosine_distance(seperated_features_test[4], centers[1])
    similarity53 = cosine_distance(seperated_features_test[4], centers[2])

    similarity61 = cosine_distance(seperated_features_test[5], centers[0])
    similarity62 = cosine_distance(seperated_features_test[5], centers[1])
    similarity63 = cosine_distance(seperated_features_test[5], centers[2])


    bins = np.linspace(0, 1, 100)
    #plt.hist(similarity11, bins, alpha=0.5, label='similarity11')
    #plt.hist(similarity22, bins, alpha=0.5, label='similarity22')
    #plt.hist(similarity31, bins, alpha=0.5, label='similarity31')
    #plt.hist(similarity32, bins, alpha=0.5, label='similarity32')

    #plt.hist(similarity41, bins, alpha=0.5, label='circleBlue-triangleBlue')
    #plt.hist(similarity42, bins, alpha=0.5, label='triangleRed-triangleBlue')
    #plt.hist(similarity43, bins, alpha=0.5, label='circleRed-triangleBlue')

    #plt.hist(np.abs(similarity51), bins, alpha=0.5, label='circleBlue-circleGreen')
    #plt.hist(np.abs(similarity61), bins, alpha=0.5, label='circleBlue-triangleGreen')

    #plt.hist(np.abs(similarity52), bins, alpha=0.5, label='triangleRed-circleGreen')
    #plt.hist(np.abs(similarity62), bins, alpha=0.5, label='triangleRed-triangleGreen')

    plt.hist(similarity11, bins, alpha=0.5, label='circleBlue-circleBlue')
    plt.hist(similarity41, bins, alpha=0.5, label='circleBlue-triangleBlue')

    plt.legend(loc='upper right')
    plt.show()

    distance_save_path = "D://projects//open_cross_entropy//save//toy_model_distance3"
    with open(distance_save_path, "wb") as f:
        pickle.dump((similarity11, similarity41, similarity51, similarity61, 
                     similarity22, similarity42, similarity52, similarity62,
                     similarity33, similarity43, similarity53, similarity63), f)