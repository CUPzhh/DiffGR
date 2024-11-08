from utils.setup_elements import input_size_match
from utils import name_match #import update_methods, retrieve_methods
from utils.utils import maybe_cuda
import torch
from utils.buffer.buffer_utils import BufferClassTracker
from utils.setup_elements import n_classes

import logging

log_file = "/media/lthpc/hd_auto/Liu/zhanghaohao/online-continual-learning-main/Log/training_log96.log"
# log_file = "/root/autodl-tmp/LOG/log1.log"

def setup_logging(log_file):
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

setup_logging(log_file)

from utils.buffer.reservoir_update import Reservoir_update

class Buffer(torch.nn.Module):
    def __init__(self, model, params):
        super().__init__()
        self.params = params
        self.model = model
        self.cuda = self.params.cuda
        self.current_index = 0
        self.n_seen_so_far = 0
        self.device = "cuda" if self.params.cuda else "cpu"
        # self.update_method = Reservoir_update(self.model, params)


        buffer_size = params.mem_size
        logging.info('buffer has %d slots' % buffer_size)
        print('buffer has %d slots' % buffer_size)
        input_size = input_size_match[params.data]
        buffer_img = maybe_cuda(torch.FloatTensor(buffer_size, *input_size).fill_(0))
        buffer_label = maybe_cuda(torch.LongTensor(buffer_size).fill_(0))

        self.register_buffer('buffer_img', buffer_img)
        self.register_buffer('buffer_label', buffer_label)

        self.update_method = name_match.update_methods[self.params.update](self.model, self.params)
        self.retrieve_method = name_match.retrieve_methods[params.retrieve](params)

        if self.params.buffer_tracker:
            self.buffer_tracker = BufferClassTracker(n_classes[params.data], self.device)

    # def update(self, x, y, **kwargs):
    #     return self.update_method.update(buffer=self, x=x, y=y, **kwargs)
    #
    # def retrieve(self, **kwargs):
    #     return self.retrieve_method.retrieve(buffer=self, **kwargs)
    #
    # def get_cls_info(self):
    #     return self.update_method.get_cls_info()
    def update(self, y,**kwargs):
        return self.update_method.update(buffer=self,  y=y, **kwargs)


    def retrieve(self, **kwargs):
        return self.retrieve_method.retrieve(buffer=self, **kwargs)