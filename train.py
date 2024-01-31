# COMP9444 Project Training File

# Running python3 train.py will train all models.

from project_requirements import *
from models import model_chooser

def label_class(s):
    if s == 'pedestrian':
        return 1
    elif s == 'rider':
        return 2
    else:
        return None

def class_label(n):
    if n == 1:
        return "pedestrian"
    elif n == 2:
        return "rider"
    else:
        return ""

class CityPersonsDataset(Dataset):
    def __init__(self, mode, transform=None):
        self.mode = mode 
        self.transform = transform
        self.dataset = glob.glob(LABEL_PATH + mode + "/*/*.json")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        with open(self.dataset[idx]) as file:
            label = json.load(file)
        img_name = self.dataset[idx].replace('CityPersons_labels/gtBboxCityPersons','Cityshapes_images/leftImg8bit')
        img_path = img_name.replace('gtBboxCityPersons.json','leftImg8bit.png')
        image = Image.open(img_path).convert("RGB")
        transform = transforms.Compose([transforms.ToTensor()])
        image = transform(image)
        
        boxes = []
        labels = []
        for o in label['objects']:
            if o['label'] in ["pedestrian", "rider"]:
                # xmin, ymin,  xmax, ymax
                boxes.append([o['bbox'][0], o['bbox'][1], o['bbox'][0] + o['bbox'][2], o['bbox'][1] + o['bbox'][3]])
                labels.append(label_class(o['label']))

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels  
        return image, target

def project_model_main():
    models = ["frcnn1", "frcnn2", "frcnn3", "frcnn4", "frcnn1b", "frcnn2b", "frcnn3b", "frcnn4b"]

    for model_name in models:
        train_loader = torch.utils.data.DataLoader(CityPersonsDataset('train'), batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
        valid_loader = torch.utils.data.DataLoader(CityPersonsDataset('val'), batch_size=2, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
        model, optimizer = model_chooser(model_name)
        model.to(DEVICE)
        num_epochs = 10
        train_stats = []
        valid_stats = []

        for epoch in range(num_epochs):
            train_stats.append([])
            valid_stats.append([])
            #train
            for i, (images,targets) in enumerate(train_loader):  
                optimizer.zero_grad()
                images = list(image.to(DEVICE) for image in images)
                targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                loss_value = losses.item()
                losses.backward()
                optimizer.step()
                print(f'Training {model_name}: Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss_value:.4f}')
                train_stats[epoch].append(loss_value)
            #validate
            for i, (images,targets) in enumerate(valid_loader):  
                images = list(image.to(DEVICE) for image in images)
                targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
                with torch.no_grad():
                    loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                loss_value = losses.item()
                print(f'Validating {model_name}: Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(valid_loader)}], Loss: {loss_value:.4f}')
                valid_stats[epoch].append(loss_value)

        torch.save(model.state_dict(), DIR + f'/{model_name}.pth')
        
        #with open(f"{model_name}_train_loss.csv", 'w') as file:
        #    file.write("epoch,step,loss\n")
        #    for ep, ep_list in enumerate(train_stats):
        #        for st, l in enumerate(ep_list):
        #            file.write(f"{ep+1},{st+1},{l}\n")
        #with open(f"{model_name}_valid_loss.csv", 'w') as file:
        #    file.write("epoch,step,loss\n")
        #    for ep, ep_list in enumerate(train_stats):
        #        for st, l in enumerate(ep_list):
        #            file.write(f"{ep+1},{st+1},{l}\n")
        
        # Plot Statistics (Loss per Epoch)
        plt.figure(figsize=(8,5))
        x = [i for i in range(1, num_epochs+1)]
        y1 = [sum(l) for l in train_stats]
        y2 = [sum(l) for l in valid_stats]
        plt.title("Loss throughout Training and Validation")
        plt.ylim(0,0.8)
        plt.plot(x, y1, label="Training")
        plt.plot(x, y2, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"{model_name}_loss.png",format='png')

if __name__ == "__main__":
    project_model_main()