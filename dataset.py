from PIL import Image
from torch.utils.data import Dataset
from os import listdir
from os.path import join
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
       

class FaceDataset(Dataset):
    def __init__(self, image_dir, upscale_factor, mode):
        super(FaceDataset, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_files = [join(image_dir, x) for x in listdir(image_dir)]
        if mode == 'train':
            self.image_files = self.image_files[0:12000]
        elif mode == 'val':
            self.image_files = self.image_files[12000:15000]
        elif mode == 'test':
            self.image_files = self.image_files[15000:]
        self.input_transform = Compose([CenterCrop(999), Resize(333), ToTensor()])
        self.target_transform = Compose([CenterCrop(999), ToTensor()])
        
    def __getitem__(self, index):
        filepath = self.image_files[index]
        try:
            input_img = Image.open(filepath).convert('RGB')
        except:
            print(self.image_files[0])
           # print(filepath)
            print(index)
            exit()
        ground_img = input_img.copy()
        input_img = self.input_transform(input_img)
        ground_img = self.target_transform(ground_img)
        return input_img, ground_img
    
    def __len__(self):
        return len(self.image_files)
