import torch
import os.path
from utils.utils import maybe_cuda

class Reservoir_update(object):
    def __init__(self, model, params):
        super().__init__()
        self.model = model
        self.params = params


    def update(self, buffer,  y, **kwargs):
        batch_size = y.size(0)  # 获取当前批次的大小

        # 计算 buffer 中剩余的空间，如果未填满，则逐个添加
        place_left = max(0, buffer.buffer_img.size(0) - buffer.current_index)
        if place_left:
            offset = min(place_left, batch_size)
            buffer.buffer_label[buffer.current_index: buffer.current_index + offset].data.copy_(y[:offset])

            # 将索引向右移
            buffer.current_index += offset
            buffer.n_seen_so_far += offset

            if offset == y.size(0):  # 如果所有数据都已添加，则返回
                filled_idx = list(range(buffer.current_index - offset, buffer.current_index))
                if buffer.params.buffer_tracker:
                    buffer.buffer_tracker.update_cache(buffer.buffer_label, y[:offset], filled_idx)
                return filled_idx

        # 如果 buffer 已经填满，则进行蓄水池采样
        y = y[place_left:]

        # 从 0 到 buffer.n_seen_so_far 的均匀分布中随机采样
        indices = torch.FloatTensor(y.size(0)).to(y.device).uniform_(0, buffer.n_seen_so_far).long()
        valid_indices = (indices < buffer.buffer_label.size(0)).long()  # 使用 buffer_label 的大小进行检查

        # 找出 valid_indices 中非零项，得到有效的 indices
        idx_new_data = valid_indices.nonzero().squeeze(-1)
        idx_buffer = indices[idx_new_data]

        buffer.n_seen_so_far += y.size(0)  # 更新已见样本数

        if idx_buffer.numel() == 0:
            return []

        assert idx_buffer.max() < buffer.buffer_img.size(0)
        assert idx_buffer.max() < buffer.buffer_label.size(0)
        assert idx_new_data.max() < y.size(0)

        # 生成索引映射字典，记录新的数据替换的位置
        idx_map = {idx_buffer[i].item(): idx_new_data[i].item() for i in range(idx_buffer.size(0))}

        replace_y = y[list(idx_map.values())]
        if buffer.params.buffer_tracker:
            buffer.buffer_tracker.update_cache(buffer.buffer_label, replace_y, list(idx_map.keys()))

        # 执行替换操作
        buffer.buffer_label[list(idx_map.keys())] = replace_y

        return list(idx_map.keys())

    # def update_cls_info(self, x, y):
    #     for label in y.unique():  # 遍历 y 中的每个唯一标签
    #         label_item = label.item()
    #         label_mask = (y == label)  # 创建一个布尔掩码来选择 x 和 y 中属于当前标签的数据
    #
    #         features = []
    #         for ex in x[label_mask]:  # 遍历属于当前标签的数据，计算特征并归一化
    #             feature = self.model.features(ex.unsqueeze(0)).detach().clone()
    #             feature = feature.squeeze()
    #             feature.data = feature.data / feature.data.norm()  # 归一化特征向量
    #             features.append(feature)
    #
    #         features = torch.stack(features)
    #         mu_y = features.mean(0).squeeze()
    #
    #         # 对 mu_y 进行归一化处理
    #         mu_y.data = mu_y.data / mu_y.data.norm()
    #
    #         if label_item not in self.cls_info:  # 如果 cls_info 中还没有该标签的信息，初始化它
    #             self.cls_info[label_item] = {
    #                 'mean': mu_y,
    #                 'count': label_mask.sum().item()}
    #         else:
    #             current_mean = self.cls_info[label_item]['mean']
    #             current_count = self.cls_info[label_item]['count']
    #             new_count = current_count + label_mask.sum().item()  # 更新后的计数
    #
    #             # 计算更新后的类心
    #             updated_mean = (current_mean * current_count + mu_y * label_mask.sum().item()) / new_count
    #             updated_mean = updated_mean/updated_mean.norm()
    #
    #             # 更新 cls_info 中的信息
    #             self.cls_info[label_item]['mean'] = updated_mean
    #             self.cls_info[label_item]['count'] = new_count
    #
    # def get_cls_info(self):
    #     return self.cls_info
