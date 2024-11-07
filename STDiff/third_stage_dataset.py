import torch
import torch.utils.data as data
import glob
import os

class ThirdStageDataset(data.Dataset):
    def __init__(self,root,split="train",transform=None):
        super(ThirdStageDataset, self).__init__()
        self.data=[]
        for f in glob.glob(os.path.join(root,split,"*.csv")):
            self.data.append(torch.load(f))

    def __getitem__(self, index):
        # # mask hot, cold, windy, mixed
        # mask = torch.ones((self.num_classes), device=self.device)
        # mask[5]=0
        # mask[7]=0
        # mask[8]=0
        # mask[-1]=0
        return self.data[index]["img"], self.data[index]["latent"], self.data[index]["w"], self.data[index]["wlabel_nxt"], self.data[index]["flabel"], self.data[index]["flabel_nxt"], self.data[index]["t"]

    def __len__(self):
        return len(self.data)
