import torch
from Toy_model import toy_model
from Toy_dataset import toy_dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import SGD
import pickle
from sklearn.metrics import confusion_matrix
from plot_utils import plot_confusion_matrix


if __name__ == "__main__":
    
    batch_size = 64
    data_path = "D://projects//open_cross_entropy//code//toy_data"
    test_data_path = "D://projects//open_cross_entropy//code//toy_data_test_inliers"
    model_path = "D://projects//open_cross_entropy//save//toy_model_E2"
    losses_path = "D://projects//open_cross_entropy//save//losses_model_E2"

    label_mapping = {"circle": 0, "rectangle": 1, "circleRed": 2}               # "circleGreen", "circleRed": 2
    num_classes = len(label_mapping)
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.RandomHorizontalFlip(),
                                         ])      # transforms.Normalize(mean=(122.15, 94.52, 127.69), std=(127.39, 123.16, 127.50))

    dataset = toy_dataset(data_path, label_mapping, data_transform)
    dataset_test = toy_dataset(data_path, label_mapping, data_transform)
    print(len(dataset))
    data_loader = DataLoader(dataset, batch_size, num_workers=4, shuffle=True)
    test_data_loader = DataLoader(dataset_test, 1, num_workers=4, shuffle=True)

    epochs = 10
    lr = 1e-8
    model = toy_model(num_classes)
    criteria = torch.nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr)

    loss_best = 1e10
    acc_best = -1e10
    losses = []
    accs = []
    confusions = []
    for e in range(epochs):
        loss_epoch = 0

        for i, (x, y) in enumerate(data_loader):
      
            x = x.float()
            y = y.type(torch.LongTensor)
            x = x.permute(0, 3, 1, 2)
            pred = model(x)
            loss = criteria(pred, y)
            #print(loss.item())
            loss.backward()
            optimizer.step()

            loss_epoch += loss.item()

        if loss_epoch < loss_best:
            #torch.save(model.state_dict(), model_path) 
            loss_best = loss_epoch
        
        losses.append(loss_epoch)
        print("epoch", e, "loss is", loss_epoch/len(dataset))
        loss_epoch = 0

        unequals = 0
        preds = []
        actuals = []
        for i, (x, y) in enumerate(test_data_loader):

            x = x.float()
            x = x.permute(0, 3, 1, 2)
            pred = model(x)
            pred = torch.argmax(pred)
            #print(pred, y)
            preds.append(pred.item())
            actuals.append(y.item())
            if pred.item() != y.item():
                unequals += 1

        conf_matrix = confusion_matrix(preds, actuals)
        confusions.append(conf_matrix)
        #plot_confusion_matrix(conf_matrix, "D://projects//open_cross_entropy//save//confusion_class3_" + str(e) + ".png")

        acc = 1-unequals*1.0 / len(dataset_test)
        print("testing accuracy is ", acc)
        accs.append(acc)
        if acc > acc_best:
            #torch.save(model.state_dict(), model_path) 
            acc_best = acc
        
        model_path_epoch = model_path + "_" + str(e)
        torch.save(model.state_dict(), model_path_epoch)

    print("best loss: ", loss_best, "best acc: ", acc_best)
    with open(losses_path, "wb") as f:
        pickle.dump((losses, accs, confusions), f)