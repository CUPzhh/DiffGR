import torch
from torch.utils import data
from utils.buffer.buffer import Buffer
from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform
from utils.setup_elements import transforms_match, input_size_match
from utils.utils import maybe_cuda, AverageMeter
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale
import torch.nn as nn
import numpy as np
from PIL import Image

import cv2
import os
import pickle
from diffusers import StableDiffusionPipeline, LCMScheduler, AutoPipelineForText2Image

# 创建自定义安全检查函数
def dummy_safety_checker(images, clip_input):
    return images, [False] * len(images)

model_id = "/media/lthpc/hd_auto/Liu/zhanghaohao/diffusion_model/stable-diffusion-v1-5/"
adapter_id = "/media/lthpc/hd_auto/Liu/zhanghaohao/LCM/lora/lcm-lora-sdv1-5/"

pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.safety_checker = dummy_safety_checker
pipe.to("cuda")

pipe.load_lora_weights(adapter_id)
pipe.fuse_lora()

out_root = "/media/lthpc/hd_auto/Liu/zhanghaohao/diffusion_model/vis/{}"
num_inference_steps = 2

with open("/media/lthpc/hd_auto/Liu/zhanghaohao/lianxi/meta", 'rb') as f:
    info = pickle.load(f)
    fl = info['fine_label_names']

import logging
# 配置日志记录
logging.basicConfig(filename="/media/lthpc/hd_auto/Liu/zhanghaohao/online-continual-learning-main/Log/training_log42.log", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SupContrastReplay(ContinualLearner):
    def __init__(self, model, opt, params):
        super(SupContrastReplay, self).__init__(model, opt, params)
        self.buffer = Buffer(model, params)
        self.mem_size = params.mem_size
        self.eps_mem_batch = params.eps_mem_batch
        self.mem_iters = params.mem_iters
        self.transform = nn.Sequential(
            RandomResizedCrop(size=(input_size_match[self.params.data][1], input_size_match[self.params.data][2]), scale=(0.2, 1.)),
            RandomHorizontalFlip(),
            ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
            RandomGrayscale(p=0.2)
        )
        self.cls_info = {}

    def gen_img(self, target_classes):
        out_imgs = []
        for target_class in target_classes:
            prompt = "a photograph of {}".format(fl[target_class.item()])
            with torch.no_grad():
                image = pipe(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=0).images[0]
                image = np.array(image)
                resized_image = cv2.resize(image, (32, 32))
                resized_image = Image.fromarray(resized_image)
                resized_image = np.array(resized_image)
                resized_image = np.transpose(resized_image, (2, 0, 1))
                out_imgs.append(resized_image)

        if out_imgs:
            out_numpy = np.stack(out_imgs)
            out_tensor = torch.tensor(out_numpy).float() / 255.0
            return out_tensor

    def save_checkpoint(self, epoch, loss, save_dir="/media/lthpc/hd_auto/Liu/zhanghaohao/online-continual-learning-main/checkpoints/checkpoint9/", filename='checkpoint.pth.tar'):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        filepath = os.path.join(save_dir, filename)
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.opt.state_dict(),
            'loss': loss,
        }
        torch.save(state, filepath)
        print(f"Checkpoint saved to {filepath}")
        logging.info(f"Checkpoint saved to {filepath}")

    def update_cls_info(self, x, y):
        with torch.no_grad():
            for label in y.unique():
                label_item = label.item()
                label_mask = (y == label)

                features = []
                for ex in x[label_mask]:
                    feature = self.model.features(ex.unsqueeze(0)).detach().clone()
                    feature = feature.squeeze()
                    feature.data = feature.data / feature.data.norm()
                    features.append(feature)

                features = torch.stack(features)
                mu_y = features.mean(0).squeeze()
                mu_y = mu_y / mu_y.norm()

                if label_item not in self.cls_info:
                    self.cls_info[label_item] = {
                        'mean': mu_y,
                        'count': label_mask.sum().item()}
                else:
                    current_mean = self.cls_info[label_item]['mean']
                    current_count = self.cls_info[label_item]['count']
                    new_count = current_count + label_mask.sum().item()
                    updated_mean = (current_mean * current_count + mu_y * label_mask.sum().item()) / new_count
                    updated_mean = updated_mean / updated_mean.norm()
                    self.cls_info[label_item]['mean'] = updated_mean
                    self.cls_info[label_item]['count'] = new_count

    def update_cls_info_with_combined(self, combined_batch, combined_labels):
        self.update_cls_info(combined_batch, combined_labels)

    def get_exemplar_means(self):
        return {cls: self.cls_info[cls]['mean'] for cls in self.cls_info}

    def train_learner(self, x_train, y_train):
        self.before_train(x_train, y_train)
        train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[self.data])
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch, shuffle=True, num_workers=0, drop_last=True)
        self.model = self.model.train()
        losses = AverageMeter()
        acc_batch = AverageMeter()

        # for ep in range(self.epoch):
        for ep in range(10):
            for i, batch_data in enumerate(train_loader):
                batch_x, batch_y = batch_data
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_y = maybe_cuda(batch_y, self.cuda)

                for j in range(self.mem_iters):
                    # if i <= 9 or i % 10 == 0:
                    mem_y = self.buffer.retrieve(y=batch_y)
                    mem_x = self.gen_img(mem_y)

                    if mem_x is not None and mem_x.size(0) > 0:
                        mem_x = maybe_cuda(mem_x, self.cuda)
                        mem_y = maybe_cuda(mem_y, self.cuda)
                        combined_batch = torch.cat((mem_x, batch_x))
                        combined_labels = torch.cat((mem_y, batch_y))
                        combined_batch_aug = self.transform(combined_batch)
                        features = torch.cat([self.model.forward(combined_batch).unsqueeze(1), self.model.forward(combined_batch_aug).unsqueeze(1)], dim=1)
                        loss = self.criterion(features, combined_labels)
                        losses.update(loss, batch_y.size(0))
                        self.opt.zero_grad()
                        loss.backward()
                        self.opt.step()
                    else:
                        combined_batch = batch_x
                        combined_labels = batch_y

                    self.update_cls_info_with_combined(combined_batch, combined_labels)
                self.update_cls_info_with_combined(combined_batch, combined_labels)

                self.buffer.update(batch_y)  # 只传递 batch_y

                if i % 100 == 0 and self.verbose:
                    print(logging.info('==>>> it: {}, avg. loss: {:.6f}, '.format(i, losses.avg(), acc_batch.avg())))

            self.save_checkpoint(ep, losses.avg(), save_dir='checkpoints', filename=f'checkpoint_epoch_{ep}.pth.tar')
        self.after_train()
