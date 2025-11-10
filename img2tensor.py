import torchvision.transforms as transforms
from PIL import Image
import os
import torchvision
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

device = 'cuda'
resnet18 = torchvision.models.resnet18(pretrained=True).to(device)
resnet18.eval()
resnet18.requires_grad_(False)
resNet18Layer4 = resnet18._modules.get('layer4').to(device)

def get_vector(t_img):
    t_img = Variable(t_img)
    my_embedding = torch.zeros(1, 512, 7, 7)
    def copy_data(m, i, o):
        my_embedding.copy_(o.data)
    h = resNet18Layer4.register_forward_hook(copy_data)
    resnet18(t_img)
    h.remove()
    return my_embedding

class extractImageFeatureResNetDataSet():
    def __init__(self):
        self.data = os.listdir('dataset/augmented')
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name = self.data[idx]
        img_loc = 'dataset/augmented/'+str(image_name)
        img = Image.open(img_loc)
        t_img = self.normalize(self.to_tensor(img))
        t_img = t_img.to(device)
        t_img = get_vector(t_img.unsqueeze(0))
        t_img = t_img.squeeze(0)
        return image_name, t_img

ImageDataset_ResNet = extractImageFeatureResNetDataSet()
resnet_loader = DataLoader(ImageDataset_ResNet, batch_size=512)

for name, embedding in tqdm(resnet_loader):
    for i in range(embedding.shape[0]):
        torch.save(embedding[i].clone(), 'tensor/{}.pt'.format(name[i][:-4]))