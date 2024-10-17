import numpy as np
from torchvision import datasets


class ExpertImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        file_name = path.split("\\")[-1]
        action_file_name = file_name.split("-")[0]
        action_index = file_name.split(".")[0]
        action_index = int(action_index.split("-")[-1])
        action = np.load("data/expert/" + action_file_name + ".npy")[action_index]

        return sample, target, action
