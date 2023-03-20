import pickle
import torch
from torch.utils.data import DataLoader
from Toy_dataset import toy_dataset
from Toy_model import toy_model
import torchvision.transforms as transforms


def normalFeatureReading(model, feature_save_path):
    
    outputs = []
    labels = []
    
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            #print("hook working!!!", name, output.shape)
            activation[name] = output.detach()
        return hook
    
    # TODO loop through the layers and register hook to the specific layer
    # https://zhuanlan.zhihu.com/p/87853615

    for name, module in model.named_modules():
        print(name)
        module.register_forward_hook(get_activation(name))

    data_transform = transforms.Compose([transforms.ToPILImage(),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor()])
    data_path = "D://projects//open_cross_entropy//code//toy_data_test_outliers"
    label_mapping = {"circle": 0, "rectangle": 1, "circleRed": 2,  "rectangleBlue": 3}                                 #  "circleRed": 2, "rectangleBlue": 3, "circleGreen": 4, "rectangleGreen": 5
    dataset = toy_dataset(data_path, label_mapping, data_transform)

    print(len(dataset))
    data_loader = DataLoader(dataset, batch_size = 1, 
                             num_workers=12, shuffle = True)
    print(len(data_loader))

    for i, (img, label) in enumerate(data_loader):
        
        print(i)
        img = img.float()
        img = img.permute(0, 3, 1, 2)
        activation = {}
        hook_output = model(img)
        outputs.append(activation)                            
        labels.append(label)


    with open(feature_save_path, "wb") as f:
        pickle.dump((outputs, labels), f)


if __name__ == "__main__":

    model = toy_model(num_classes=3)
    model_path = "D://projects//open_cross_entropy//save//toy_model3_E2_999"
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    feature_save_path = "D://projects//open_cross_entropy//save//toy_model_E2_999"
    normalFeatureReading(model, feature_save_path)
    