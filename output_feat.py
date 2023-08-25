import torch
import glob
from torchvision import transforms
# npy dict struct:
from PIL import Image
from tqdm import tqdm
import numpy as np
"""
{
    scale0: array,
    scale1: array,
    scale2: array,
}
"""
from resnet import resnet18, resnet34, resnet50, wide_resnet50_2


def get_slice_boxes(image_height, image_width, slice_size, overlap_ratio):
    slice_bboxes = []
    y_max = y_min = 0

    y_overlap = int(overlap_ratio * slice_size)
    x_overlap = int(overlap_ratio * slice_size)

    while y_max < image_height:
        x_min = x_max = 0
        y_max = y_min + slice_size
        while x_max < image_width:
            x_max = x_min + slice_size
            if y_max > image_height or x_max > image_width:
                xmax = min(image_width, x_max)
                ymax = min(image_height, y_max)
                xmin = max(0, xmax - slice_size)
                ymin = max(0, ymax - slice_size)
                slice_bboxes.append([ymin, xmin, ymax, xmax])
            else:
                slice_bboxes.append([y_min, x_min, y_max, x_max])
            x_min = x_max - x_overlap
        y_min = y_max - y_overlap
    return slice_bboxes

def merge_feat(feats, image_height, image_width, slice_size, overlap_ratio):
    slice_boxes = get_slice_boxes(image_height, image_width, slice_size, overlap_ratio)

    final_feat = np.zeros((feats[0].shape[0], image_height, image_width))
    count_map = np.zeros((image_height, image_width))

    for box in slice_boxes:
        count_map[box[0]:box[2], box[1]:box[3]] += 1
    weight_map = 1 / count_map

    for feat, box in zip(feats, slice_boxes):
        final_feat[:, box[0]:box[2], box[1]:box[3]] += feat

    final_feat *= weight_map

    return final_feat

class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, overlap_ratio=0.5, slice_size=256, max_downscale=16):
        self.img_paths = glob.glob(root+'/*/*.jpg')
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.03016077, 0.03016077, 0.03016077],
                std=[0.17366856, 0.17366856, 0.17366856],
            )
        ])
        self.slice_size = slice_size
        self.overlap_ratio = overlap_ratio
        self.max_downscale = 16
    
    def __len__(self):
        return len(self.img_paths)
    
    def get_sliced_imgs(self, img, slice_boxes):
        sliced_imgs = []
        for box in slice_boxes:
            sliced_img = img[:, box[0]: box[2], box[1]:box[3]]
            sliced_imgs.append(sliced_img)
        return sliced_imgs
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')

        resize_h = (img.height // self.max_downscale) * self.max_downscale
        resize_w = (img.width // self.max_downscale) * self.max_downscale
        img = img.resize((resize_w, resize_h), Image.Resampling.BILINEAR)

        img_tensor = self.transforms(img)
        slice_boxes = get_slice_boxes(img.height, img.width, self.slice_size, self.overlap_ratio)
        sliced_img_tensors = self.get_sliced_imgs(img_tensor, slice_boxes)
        return sliced_img_tensors, torch.tensor([img.height, img.width]), img_path


def output_feat(root, batch_size):
    dataset = Dataset(root, 0.5, 256)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    encoder, _ = wide_resnet50_2(pretrained=True)
    encoder.cuda()
    encoder.eval()
    for imgs, orig_shape, img_path in tqdm(dataloader):
        orig_shape = orig_shape[0] #bs 1 squeeze
        print(orig_shape)
        feats_scale0 = []
        feats_scale1 = []
        feats_scale2 = []
        imgs_len = len(imgs)
        batches_inds = list(map(lambda a: [a, a+batch_size] if a+batch_size < imgs_len else [a, imgs_len],  list(range(0, imgs_len, batch_size))))
        for b_inds in batches_inds:
            img_batch = torch.cat(imgs[b_inds[0]:b_inds[1]], dim=0)
            img_batch = img_batch.cuda()
            # print(img_batch.shape)
            with torch.no_grad():
                batched_feats = encoder(img_batch)
                batched_feat0 = batched_feats[0].detach().cpu().numpy()
                batched_feat1 = batched_feats[1].detach().cpu().numpy()
                batched_feat2 = batched_feats[2].detach().cpu().numpy()
            feats_scale0.extend([batched_feat0[i] for i in range(len(batched_feat0))])
            feats_scale1.extend([batched_feat1[i] for i in range(len(batched_feat1))])
            feats_scale2.extend([batched_feat2[i] for i in range(len(batched_feat2))])
        merged_feat0 = merge_feat(feats_scale0, orig_shape[0] // 4, orig_shape[1] // 4, 64, 0.5)
        merged_feat1 = merge_feat(feats_scale1, orig_shape[0] // 8, orig_shape[1] // 8, 32, 0.5)
        merged_feat2 = merge_feat(feats_scale2, orig_shape[0] // 16, orig_shape[1] // 16, 16, 0.5)
        saved = {
            "truncted_shape": orig_shape,
            "feat0": merged_feat0,
            "feat1": merged_feat1,
            "feat2": merged_feat2,
        }
        np.save(img_path[0].replace('.jpg', '.npy'), saved, allow_pickle=True)

output_feat('/home/zhougaowei/datasets/xray/mvtec/cans/train', 4)
output_feat('/home/zhougaowei/datasets/xray/mvtec/cans/test', 4)