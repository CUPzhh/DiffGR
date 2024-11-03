import torch
from torch.utils import data
from utils.buffer.buffer import Buffer
from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform
from utils.setup_elements import transforms_match, input_size_match
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale
import torch.nn as nn
from utils.setup_elements import transforms_match
from utils.utils import maybe_cuda, AverageMeter

import numpy as np
from PIL import Image
import cv2
import os
import pickle
from diffusers import StableDiffusionPipeline, LCMScheduler

# 创建自定义安全检查函数
def dummy_safety_checker(images, clip_input):
    return images, [False] * len(images)

import logging
# 配置日志记录

logging.basicConfig(filename="/media/lthpc/hd_auto/Liu/zhanghaohao/online-continual-learning-main/Log/training_log96.log",
                    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

model_id = "/media/lthpc/hd_auto/Liu/zhanghaohao/diffusion_model/stable-diffusion-v1-5/"
adapter_id = "/media/lthpc/hd_auto/Liu/zhanghaohao/LCM/lora/lcm-lora-sdv1-5/"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.safety_checker = dummy_safety_checker
pipe.to("cuda")

pipe.load_lora_weights(adapter_id)
pipe.fuse_lora()

out_root = "/media/lthpc/hd_auto/Liu/zhanghaohao/diffusion_model/vis/STDv1-5/"
os.makedirs(out_root, exist_ok=True)
num_inference_steps = 2

with open("/media/lthpc/hd_auto/Liu/zhanghaohao/lianxi/meta", 'rb') as f:
    info = pickle.load(f)
    fl = info['fine_label_names']

def calculate_accuracy(logits, targets, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = targets.size(0)

    _, pred = logits.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class ExperienceReplay(ContinualLearner):
    def __init__(self, model, opt, params):
        super(ExperienceReplay, self).__init__(model, opt, params)
        self.buffer = Buffer(model, params)
        self.mem_size = params.mem_size
        self.eps_mem_batch = params.eps_mem_batch
        self.mem_iters = params.mem_iters
        self.transform = nn.Sequential(
            RandomResizedCrop(size=(input_size_match[self.params.data][1], input_size_match[self.params.data][2]),
                              scale=(0.2, 1.)),
            RandomHorizontalFlip(),
            ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
            RandomGrayscale(p=0.2))

    def gen_img(self, target_classes):
        # 创建嵌套字典
        categories = {
            "aquatic mammals": {4: "beaver", 30: "dolphin", 55: "otter", 72: "seal", 95: "whale"},
            "fish": {1: "aquarium_fish", 32: "flatfish", 67: "ray", 73: "shark", 91: "trout"},
            "flowers": {54: "orchids", 62: "poppies", 70: "roses", 82: "sunflowers", 92: "tulips"},
            "food containers": {9: "bottle", 10: "bowl", 16: "can", 28: "cup", 61: "plate"},
            "fruit_and_vegetables": {0: "apples", 51: "mushrooms", 53: "oranges", 57: "pears", 83: "sweet_peppers"},
            "household appliances": {22: "clock", 39: "keyboard", 40: "lamp", 86: "telephone", 87: "television"},
            "household furniture": {5: "bed", 20: "chair", 25: "couch", 84: "table", 94: "wardrobe"},
            "insects": {6: "bee", 7: "beetle", 14: "butterfly", 18: "caterpillar", 24: "cockroach"},
            "large carnivores": {3: "bear", 42: "leopard", 43: "lion", 88: "tiger", 97: "wolf"},
            "large outdoor man-made objects": {12: "bridge", 17: "castle", 37: "house", 68: "road", 76: "skyscraper"},
            "large outdoor nature landscape": {23: "cloud", 33: "forest", 49: "mountain", 60: "plain", 71: "sea"},
            "large omnivores and herbivores": {15: "camel", 19: "cattle", 21: "chimpanzee", 31: "elephant",
                                               38: "kangaroo"},
            "medium mammals": {34: "fox", 63: "porcupine", 64: "possum", 66: "raccoon", 75: "skunk"},
            "non-insect invertebrates": {26: "crab", 45: "lobster", 77: "snail", 79: "spider", 99: "worm"},
            "people": {2: "baby", 11: "boy", 35: "girl", 46: "man", 98: "woman"},
            "reptiles": {27: "crocodile", 29: "dinosaur", 44: "lizard", 78: "snake", 93: "turtle"},
            "small mammals": {36: "hamster", 50: "mouse", 65: "rabbit", 74: "shrew", 80: "squirrel"},
            "trees": {47: "maple_tree", 52: "oak_tree", 56: "palm_tree", 59: "pine_tree", 96: "willow_tree"},
            "vehicles_1": {8: "bicycle", 13: "bus", 48: "motorcycle", 58: "pickup_truck", 90: "train"},
            "vehicles_2": {41: "lawn_mower", 69: "rocket", 81: "streetcar", 85: "tank", 89: "tractor"}
        }

        out_imgs = []
        for target_class in target_classes:
            # 遍历大类以找到对应的子类和大类
            category_name, group_name = None, None
            for group, items in categories.items():
                if target_class.item() in items:
                    category_name = items[target_class.item()]
                    group_name = group
                    break

            if category_name and group_name:
                # 生成带有大类名称的 prompt
                prompt = "a photograph of {}, {}".format(category_name, group_name)
                negative_prompt = "3d, cartoon, low_quality, deforme"
                with torch.no_grad():
                    image = pipe(prompt=prompt, negative_prompt=negative_prompt,
                                 num_inference_steps=num_inference_steps,
                                 guidance_scale=0,
                                 ).images[0]
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

    def train_learner(self, x_train, y_train):
        self.before_train(x_train, y_train)
        # set up loader
        train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[self.data])
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch, shuffle=True, num_workers=0,
                                       drop_last=True)
        # set up model
        self.model = self.model.train()

        # setup tracker
        losses_batch = AverageMeter()
        losses_mem = AverageMeter()
        acc1_batch = AverageMeter()
        acc5_batch = AverageMeter()
        acc1_mem = AverageMeter()
        acc5_mem = AverageMeter()

        for ep in range(self.epoch):
            for i, batch_data in enumerate(train_loader):
                # batch update
                batch_x, batch_y = batch_data
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_y = maybe_cuda(batch_y, self.cuda)
                for j in range(self.mem_iters):

                    # 对 batch_x 进行数据增强
                    augmented_batch_x = self.transform(batch_x)

                    # 将增强后的数据与原始数据拼接
                    combined_batch_x = torch.cat((batch_x, augmented_batch_x), dim=0)

                    # forward 计算 logits
                    logits = self.model.forward(combined_batch_x)

                    # 对 logits 和 batch_y 进行匹配处理
                    combined_batch_y = torch.cat((batch_y, batch_y), dim=0)

                    # 计算 loss
                    loss = self.criterion(logits, combined_batch_y)
                    if self.params.trick['kd_trick']:
                        loss = 1 / (self.task_seen + 1) * loss + (1 - 1 / (self.task_seen + 1)) * \
                                   self.kd_manager.get_kd_loss(logits, batch_x)
                    if self.params.trick['kd_trick_star']:
                        loss = 1/((self.task_seen + 1) ** 0.5) * loss + \
                               (1 - 1/((self.task_seen + 1) ** 0.5)) * self.kd_manager.get_kd_loss(logits, batch_x)
                    top1_acc, top5_acc = calculate_accuracy(logits, combined_batch_y, topk=(1, 5))
                    acc1_batch.update(top1_acc.item(), combined_batch_y.size(0))
                    acc5_batch.update(top5_acc.item(), combined_batch_y.size(0))
                    losses_batch.update(loss, batch_y.size(0))
                    # backward
                    self.opt.zero_grad()
                    loss.backward()

                    # mem update
                    # mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)

                    if i <= 9 or i % 5 == 0:
                        mem_y = self.buffer.retrieve(y=batch_y)
                        mem_x = self.gen_img(mem_y)

                    if mem_x is not None and mem_x.size(0) > 0:
                        mem_x = maybe_cuda(mem_x, self.cuda)
                        mem_y = maybe_cuda(mem_y, self.cuda)

                        # 对 mem_x 进行数据增强
                        augmented_mem_x = self.transform(mem_x)

                        # 将增强后的数据与原始 mem_x 拼接
                        combined_mem_x = torch.cat((mem_x, augmented_mem_x), dim=0)

                        # forward 计算 mem_logits
                        mem_logits = self.model.forward(combined_mem_x)

                        # 扩展 mem_y 以匹配拼接后的 mem_x
                        combined_mem_y = torch.cat((mem_y, mem_y), dim=0)

                        # 计算 loss
                        loss_mem = self.criterion(mem_logits, combined_mem_y)

                        if self.params.trick['kd_trick']:
                            loss_mem = 1 / (self.task_seen + 1) * loss_mem + (1 - 1 / (self.task_seen + 1)) * \
                                       self.kd_manager.get_kd_loss(mem_logits, combined_mem_x)

                        if self.params.trick['kd_trick_star']:
                            loss_mem = 1 / ((self.task_seen + 1) ** 0.5) * loss_mem + \
                                       (1 - 1 / ((self.task_seen + 1) ** 0.5)) * self.kd_manager.get_kd_loss(mem_logits,
                                                                                                             combined_mem_x)

                        # update tracker
                        top1_mem_acc, top5_mem_acc = calculate_accuracy(mem_logits, combined_mem_y, topk=(1, 5))
                        acc1_mem.update(top1_mem_acc.item(), combined_mem_y.size(0))
                        acc5_mem.update(top5_mem_acc.item(), combined_mem_y.size(0))
                        losses_mem.update(loss_mem, mem_y.size(0))
                        loss_mem.backward()

                    if self.params.update == 'ASER' or self.params.retrieve == 'ASER':
                        # opt update
                        self.opt.zero_grad()
                        combined_batch = torch.cat((mem_x, batch_x))
                        combined_labels = torch.cat((mem_y, batch_y))
                        combined_logits = self.model.forward(combined_batch)
                        loss_combined = self.criterion(combined_logits, combined_labels)
                        loss_combined.backward()
                        self.opt.step()
                    else:
                        self.opt.step()

                        # update mem
                        self.buffer.update(batch_y)
                        if i % 100 == 1 and self.verbose:
                            logging.info(
                                '==>>> it: {}, avg. loss: {:.6f}, '
                                'running train acc1: {:.3f}, acc5: {:.3f}'
                                    .format(i, losses_batch.avg(), acc1_batch.avg(), acc5_batch.avg())
                            )
                            logging.info(
                                '==>>> it: {}, mem avg. loss: {:.6f}, '
                                'running mem acc1: {:.3f}, acc5: {:.3f}'
                                    .format(i, losses_mem.avg(), acc1_mem.avg(), acc5_mem.avg())
                            )
                    self.after_train()