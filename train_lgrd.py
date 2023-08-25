import torch
import numpy as np
import random
import os
from torch.utils.data import DataLoader
from resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50
import torch.backends.cudnn as cudnn
import argparse
from test import evaluation, visualization, test
from torch.nn import functional as F
import glob
import torchvision
from PIL import Image
from tqdm import tqdm


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def loss_fucntion(a, b):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        #print(a[item].shape)
        #print(b[item].shape)
        #loss += 0.1*mse_loss(a[item], b[item])
        loss += torch.mean(1-cos_loss(a[item].view(a[item].shape[0],-1),
                                      b[item].view(b[item].shape[0],-1)))
    return loss

class FixedTeacherTrainDataset(torch.utils.data.Dataset):
    def __init__(self, train_root):
        # self.img_paths = sorted(glob.glob(train_root+'/*/*.jpg'))
        self.teacher_feat_paths = sorted(glob.glob(train_root+'/*/*.npy'))

    def __len__(self):
        return len(self.teacher_feat_paths)

    def __getitem__(self, idx):
        teacher_feat_path = self.teacher_feat_paths[idx]
        teacher_feats = np.load(teacher_feat_path)

        return teacher_feats

        # truncted_shape = teacher_feats['truncted_shape']
        # img_path = self.img_paths[idx]
        # img = Image.open(img_path).convert('RGB')
        # img = img.resize((truncted_shape[1], truncted_shape[0]), Image.Resampling.BILINEAR)
        # img_tensor = self.transforms(img)

        # return img_tensor, teacher_feats

class FixedTeacherTestDataset(torch.utils.data.Dataset):
    def __init__(self, test_root, gt_root):
        self.teacher_feat_paths_ng = sorted(glob.glob(test_root+'/impurity/*.npy'))
        self.teacher_feat_paths_ok = sorted(glob.glob(test_root+'/good/*.npy'))
        self.gt_paths = [os.path.join(gt_root, 'impurity', os.path.basename(p).replace('.npy', '.png')) for p in self.teacher_feat_paths_ng]
        self.len_ng = len(self.teacher_feat_paths_ng)
        self.len_ok = len(self.teacher_feat_paths_ok)
        self.gt_transform = torchvision.transforms.ToTensor()

    def __len__(self):
        return self.len_ng + self.len_ok
    
    def __getitem__(self, idx):
        if idx < self.len_ok:
            teacher_feat_path = self.teacher_feat_paths_ok[idx]
            teacher_feats = np.load(teacher_feat_path)
            mask = np.zeros(teacher_feats['truncted_shape'])

        else:
            ng_idx = idx - self.len_ok
            teacher_feat_path = self.teacher_feat_paths_ng[ng_idx]
            teacher_feats = np.load(teacher_feat_path)
            truncted_shape = teacher_feats['truncted_shape']

            mask = Image.open(self.gt_paths).resize((truncted_shape[1], truncted_shape[0]), Image.Resampling.NEAREST)
            mask = self.gt_transform(mask)

        return teacher_feats, mask

def train(_class_):
    print(_class_)
    epochs = 200
    learning_rate = 0.005
    batch_size = 16
    # image_size = 256
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    train_path = './content/' + _class_ + '/train'
    test_path = './content/' + _class_ + '/test'
    gt_path = './content/' + _class_ + '/ground_truth'
    ckp_path = './checkpoints/' + 'wres50_'+_class_+'.pth'


    train_data = FixedTeacherTrainDataset(train_path)
    test_data = FixedTeacherTestDataset(test_path, gt_path)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)

    optimizer = torch.optim.Adam(list(decoder.parameters())+list(bn.parameters()), lr=learning_rate, betas=(0.5,0.999))


    for epoch in range(epochs):
        bn.train()
        decoder.train()
        loss_list = []
        for teacher_feats in tqdm(train_dataloader):
            t_feats0 = teacher_feats['feat0'].cuda()
            t_feats1 = teacher_feats['feat1'].cuda()
            t_feats2 = teacher_feats['feat2'].cuda()

            # 为了加速训练采用resized_feat
            resized_t_feats0 = F.interpolate(t_feats0, (64, 64), align_corners=True)
            resized_t_feats1 = F.interpolate(t_feats1, (32, 32), align_corners=True)
            resized_t_feats2 = F.interpolate(t_feats2, (16, 16), align_corners=True)

            outputs = decoder(bn([resized_t_feats0, resized_t_feats1, resized_t_feats2]))

            print(outputs[0].shape)

            # 计算损失时upsample到原始尺寸
            loss = loss_fucntion(
                [t_feats0, t_feats1, t_feats2], 
                [
                    F.interpolate(outputs[0], t_feats0.shape[-2:], align_corners=True),
                    F.interpolate(outputs[1], t_feats1.shape[-2:], align_corners=True),
                    F.interpolate(outputs[2], t_feats2.shape[-2:], align_corners=True)
                ]
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, np.mean(loss_list)))
        if (epoch + 1) % 10 == 0:
            auroc_px, auroc_sp, aupro_px = evaluation(encoder, bn, decoder, test_dataloader, device)
            print('Pixel Auroc:{:.3f}, Sample Auroc{:.3f}, Pixel Aupro{:.3}'.format(auroc_px, auroc_sp, aupro_px))
            torch.save({'bn': bn.state_dict(),
                        'decoder': decoder.state_dict()}, ckp_path)
    return auroc_px, auroc_sp, aupro_px



if __name__ == '__main__':

    setup_seed(111)
    item_list = ['cans_random_crop']
    for i in item_list:
        train(i)

