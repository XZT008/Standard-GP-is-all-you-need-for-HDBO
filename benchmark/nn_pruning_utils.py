import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from tqdm import tqdm
import os


# load and modifier last layer
def load_model(model_name: str):
    if model_name == "resnet152":
        model = torchvision.models.resnet152(weights='DEFAULT')
        in_channels = model.fc.in_features
        model.fc = nn.Linear(in_channels, 200)
    elif model_name == "resnet101":
        model = torchvision.models.resnet101(weights='DEFAULT')
        in_channels = model.fc.in_features
        model.fc = nn.Linear(in_channels, 200)
    elif model_name == "densenet201":
        model = torchvision.models.densenet201(weights='DEFAULT')
        in_channels = model.classifier.in_features
        model.classifier = nn.Linear(in_channels, 200)
    else:
        raise NotImplementedError
    return model


def load_model_from_checkpoint(model_name: str):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(dir_path, "saved_models")
    if model_name == "resnet152":
        model = torchvision.models.resnet152(weights='DEFAULT')
        in_channels = model.fc.in_features
        model.fc = nn.Linear(in_channels, 200)
    elif model_name == "densenet201":
        model = torchvision.models.densenet201(weights='DEFAULT')
        in_channels = model.classifier.in_features
        model.classifier = nn.Linear(in_channels, 200)
    else:
        raise NotImplementedError

    model.load_state_dict(torch.load(os.path.join(base_dir, f"{model_name}.pt")))
    return model


def create_val_img_folder():
    '''
    This method is responsible for separating validation images into separate sub folders
    '''
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dataset_dir = os.path.join(dir_path, "tiny-imagenet-200")
    # dataset_dir = r"C:\Users\xzt\Desktop\kspace_classifier_policy\TT-HDBO\benchmark\tiny-imagenet-200"
    val_dir = os.path.join(dataset_dir, 'val')
    img_dir = os.path.join(val_dir, 'images')

    fp = open(os.path.join(val_dir, 'val_annotations.txt'), 'r')
    data = fp.readlines()
    val_img_dict = {}
    for line in data:
        words = line.split('\t')
        val_img_dict[words[0]] = words[1]
    fp.close()

    # Create folder if not present and move images into proper folders
    for img, folder in val_img_dict.items():
        newpath = (os.path.join(img_dir, folder))
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        if os.path.exists(os.path.join(img_dir, img)):
            os.rename(os.path.join(img_dir, img), os.path.join(newpath, img))


# create train, val, test dataloader
def create_dataloaders(train_bs, test_bs):
    create_val_img_folder()
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dir_path = os.path.dirname(os.path.realpath(__file__))
    root_dir = os.path.join(dir_path, "tiny-imagenet-200")
    # train_trans = [transforms.RandomHorizontalFlip(), transforms.RandomResizedCrop(224), transforms.ToTensor(), norm]
    # val_trans = [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), norm]
    train_trans = [transforms.Resize(224), transforms.ToTensor(), norm]
    val_trans = [transforms.Resize(224), transforms.ToTensor(), norm]
    all_datasets = {
        "train": datasets.ImageFolder(os.path.join(root_dir, "train"), transform=transforms.Compose(train_trans)),
        "val": datasets.ImageFolder(os.path.join(root_dir, "val", "images"), transform=transforms.Compose(val_trans)),
    }

    all_dataloaders = {
        "train": torch.utils.data.DataLoader(all_datasets["train"], train_bs, shuffle=True),
        "val": torch.utils.data.DataLoader(all_datasets["val"], test_bs, shuffle=True),
    }

    return all_dataloaders


def val(model, dataloader, device, mode='val'):
    model = model.to(device)
    model.eval()
    total_loss, total_correct, total_size = 0., 0., 0.
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc=mode):
            imgs = imgs.to(device)
            labels = labels.to(device)


            outputs = model(imgs)
            _, preds = torch.max(outputs, dim=1)
            loss = criterion(outputs, labels)


            # stats
            total_loss += loss.item() * imgs.size(0)
            correct = torch.sum(torch.where(preds == labels, 1, 0))
            total_correct += correct
            total_size += imgs.size(0)

            # print(f"loss: {loss.item()}, acc: {correct / imgs.size(0)}")
    return total_loss / total_size, total_correct / total_size


def train(model, dataloader, device):
    model = model.to(device)
    model.train()
    total_loss, total_correct, total_size = 0., 0., 0.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for imgs, labels in tqdm(dataloader, desc="Training"):
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        _, preds = torch.max(outputs, dim=1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # stats
        total_loss += loss.item() * imgs.size(0)
        correct = torch.sum(torch.where(preds == labels, 1, 0))
        total_correct += correct
        total_size += imgs.size(0)

        # print(f"loss: {loss.item()}, acc: {correct / imgs.size(0)}")

    return total_loss/total_size, total_correct/total_size


# finetune all weights, use GPU if available
def fine_tuning(model_name: str, train_bs: int, test_bs: int, epochs: int):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_name).to(device)
    dataloaders = create_dataloaders(train_bs, test_bs)

    best_val_acc = 0.0
    val_avg_loss, val_avg_acc = val(model, dataloaders["val"], device)
    print(f"Before training, val acc: {val_avg_acc}, val loss: {val_avg_loss}\n")

    for epoch in range(epochs):
        print(f"At epoch {epoch}: ")
        avg_loss, avg_acc = train(model, dataloaders["train"], device)
        print(f"Training acc: {avg_acc}, training loss: {avg_loss}")
        val_avg_loss, val_avg_acc = val(model, dataloaders["val"], device)
        print(f"Val acc: {val_avg_acc}, Val loss: {val_avg_loss}")

        # save model if better val acc achieved
        if val_avg_acc > best_val_acc:
            torch.save(model.state_dict(), r"./saved_models/"+model_name+".pt")


if __name__ == '__main__':
    # fine_tuning("densenet201", 100, 512, 20)
    # fine_tuning("resnet152", 100, 512, 20)

    resnet152 = load_model_from_checkpoint("resnet152")
    densenet201 = load_model_from_checkpoint("densenet201")
    val_dataloader = create_dataloaders(100, 512)["val"]
    device = torch.device('cuda')
    val_avg_loss, val_avg_acc = val(resnet152, val_dataloader, device)
    print(f"Resnet152 ----- Val acc: {val_avg_acc}, Val loss: {val_avg_loss}")

    val_avg_loss, val_avg_acc = val(densenet201, val_dataloader, device)
    print(f"densenet201 ----- Val acc: {val_avg_acc}, Val loss: {val_avg_loss}")



