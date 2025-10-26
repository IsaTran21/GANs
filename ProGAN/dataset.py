import glob
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from concurrent.futures import ProcessPoolExecutor
from utils import get_tqdm
import time

tqdm = get_tqdm()
class CelebDataset(Dataset):

    def __init__(self, img_path, transform=None):
        super().__init__()
        self.listImg = getImgList(img_path)
        self.img_path = img_path
        self.transform = transform

    def __len__(self):
        return len(self.listImg)

    def __getitem__(self, idx):

        idx = idx % len(self.listImg)
        img_path = self.listImg[idx]
        img = Image.open(img_path)
        if self.transform:
            return self.transform(img)
        else:
            return transforms.ToTensor(img)

def getImgList(path):
    nameList = glob.glob(f'{path}*.jpg')
    return nameList



class Dataset_transformed:
    def __init__(self, im_path, batch_sizes, max_size_i=6, crop_size = 160):
        """
        max_size_i: we want to get the image size to be 4, 8, 16,...
        The power of 2, therefore, if we want the target image of size 16, thn max_size_i = 4
        """
        assert max_size_i in range(2,10), "max_size must be the power of 2"
        self.max_size_i = max_size_i
        self.im_path = im_path
        self.crop_size = crop_size
        self.batch_sizes = batch_sizes
        self.all_data = self._data()


    def get_transform(self, size):
        transform = transforms.Compose([
            transforms.CenterCrop(self.crop_size),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5, 0.5), (0.5,0.5, 0.5))  # Scales to [-1, 1]
        ])
        data = CelebDataset(self.im_path, transform=transform)
        return data

    def _data(self):
        """
        Output: a dictionary of loaders, example:
        {4: dataloader object at 4,
        8: dataloader object at 8}
        :return:
        """
        start = time.time()
        all_sizes = list(range(2, self.max_size_i))
        NUM_WORKERS = max(1, os.cpu_count() // 2)
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            results = executor.map(self.get_transform, all_sizes)
            all_data = list(tqdm(results, total=len(all_sizes), desc="creating dataset", colour="green"))
        print(f'It takes {time.time() - start} seconds')
        # all_sizes are like 4, 8, 16, 32,...
        # Then the self.batch_sizes is a dict of {4: its batch size, 5: its batch size,...}
        # range(self.max_size_i-2) is something like: 0, 1, 2,....the indices of each size
        all_loaders = {
            2**all_sizes[i]: DataLoader(
                                        all_data[i],
                                        batch_size=self.batch_sizes[2**all_sizes[i]],
                                        shuffle=True,
                                        pin_memory=True,
                                        num_workers=NUM_WORKERS,
                                        drop_last=True
                                        )\
            for i in range(self.max_size_i-2)} # because we get the index, moreover, the Py starts at 0
        return all_loaders

if __name__ == "__main__":
    pass
