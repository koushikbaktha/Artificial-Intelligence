import torch
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler, WeightedRandomSampler, Sampler


class Nav_Dataset(dataset.Dataset):
    def __init__(self):
        self.data = np.genfromtxt('saved/training_data.csv', delimiter=',')
        self.scaler = MinMaxScaler()
        self.normalized_data = self.scaler.fit_transform(self.data) #fits and transforms
        pickle.dump(self.scaler, open("saved/scaler.pkl", "wb")) #save to normalize at inference
        
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            idx = idx.item()
        else:
            x=torch.from_numpy(np.array(self.data[idx,0:6])).type(torch.FloatTensor)
            y=torch.from_numpy(np.array(self.data[idx][-1])).type(torch.FloatTensor)
            dict1={'input':x,'label':y}
            return dict1

class Data_Loaders():
    def __init__(self, batch_size):
        
        self.nav_dataset = Nav_Dataset()
        length_of_dataset=len(self.nav_dataset)
        dataset_indices = list(range(length_of_dataset))
        np.random.shuffle(dataset_indices)
        val_split_index = int(np.floor(0.2 * length_of_dataset))
        train_idx, test_idx = dataset_indices[val_split_index:], dataset_indices[:val_split_index]
        train_sampler=SubsetRandomSampler(train_idx)
        test_sampler=SubsetRandomSampler(test_idx)
        self.train_loader = DataLoader(dataset=self.nav_dataset, shuffle=False, batch_size=batch_size, sampler=train_sampler,num_workers=2)
        self.test_loader = DataLoader(dataset=self.nav_dataset, shuffle=False, batch_size=batch_size, sampler=test_sampler,num_workers=2)
        

def main():
    batch_size = 16
    data_loaders = Data_Loaders(batch_size)
    for idx, sample in enumerate(data_loaders.train_loader):
        _, _ = sample['input'], sample['label']
    for idx, sample in enumerate(data_loaders.test_loader):
        _, _ = sample['input'], sample['label']

if __name__ == '__main__':
    main()