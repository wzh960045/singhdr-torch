import os
from abc import ABC, abstractmethod
# import tensorflow as tf


class Net(ABC):

    @abstractmethod
    def get_output(*args, **kwargs):
        pass

    @abstractmethod
    def load_param(*args, **kwargs):
        pass
    
    @abstractmethod
    def save_param(*args, **kwargs):
        pass


class BaseNet(Net):

    @abstractmethod
    def _get_output(self, *args, **kwargs):
        pass
    
    def __init__(self, scope):
        self.scope = scope
        self._template = tf.make_template(
            self.scope,
            self._get_output,
        )
        return
    
    def get_output(self, *args, **kwargs):
        return self._template(*args, **kwargs)
    
    def _get_saver(self):
        if not hasattr(self, '_saver'):
            self._saver = tf.train.Saver(list(filter(lambda a: 'Adam' not in a.name, tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES,
                scope=self.scope,
            ))), max_to_keep=None)
        return self._saver
    
    def load_param(self, sess, pretrain):
        if os.path.isdir(pretrain):
            pretrain = tf.train.latest_checkpoint(os.path.join(pretrain, self.scope))
        if pretrain:
            self._get_saver().restore(sess, pretrain)
        return
    
    def save_param(self, sess, save_dir, it):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, self.scope, 'log-%d' % it)
        self._get_saver().save(sess, save_path)
        return


class AggNet(Net):

    @abstractmethod
    def _get_output(self, *args, **kwargs):
        pass

    def __init__(self, sub_net_list):
        self.sub_net_list = sub_net_list
        return

    def get_output(self, *args, **kwargs):
        return self._get_output(*args, **kwargs)
    
    def load_param(self, sess, pretrain):
        for sub_net in self.sub_net_list:
            sub_net.load_param(sess, pretrain)
        return
    
    def save_param(self, sess, save_dir, it):
        for sub_net in self.sub_net_list:
            sub_net.save_param(sess, save_dir, it)
        return
    


import os
from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class Net(ABC, nn.Module):
    @abstractmethod
    def get_output(self, *args, **kwargs):
        pass

    @abstractmethod
    def load_param(self, *args, **kwargs):
        pass

    @abstractmethod
    def save_param(self, *args, **kwargs):
        pass


class BaseNet(Net):
    @abstractmethod
    def _get_output(self, *args, **kwargs):
        pass

    def __init__(self, scope: str):
        super().__init__()
        self.scope = scope
        # 在 PyTorch 中，forward 方法本身就起到类似模板的作用
        # 所以不需要像 tf.make_template 这样的机制

    def forward(self, *args, **kwargs):
        return self._get_output(*args, **kwargs)

    def get_output(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _get_saver(self):
        """
        模拟 Saver，这里返回自身即可，因为保存是直接调用 torch.save
        """
        return self

    def load_param(self, device, pretrain_path):
        if os.path.isdir(pretrain_path):
            # 查找最新模型文件
            model_dir = os.path.join(pretrain_path, self.scope)
            if not os.path.exists(model_dir):
                raise FileNotFoundError(f"Model directory {model_dir} does not exist.")
            model_files = [f for f in os.listdir(model_dir) if f.startswith("log-")]
            if not model_files:
                raise FileNotFoundError(f"No checkpoint found in {model_dir}")
            latest_file = sorted(model_files)[-1]
            pretrain_path = os.path.join(model_dir, latest_file)

        if pretrain_path:
            state_dict = torch.load(pretrain_path, map_location=device)
            self.load_state_dict(state_dict)
        return

    def save_param(self, save_dir, it):
        model_dir = os.path.join(save_dir, self.scope)
        os.makedirs(model_dir, exist_ok=True)
        save_path = os.path.join(model_dir, f"log-{it}.pt")
        torch.save(self.state_dict(), save_path)
        return
    


class AggNet(Net):
    @abstractmethod
    def _get_output(self, *args, **kwargs):
        pass

    def __init__(self, sub_net_list):
        self.sub_net_list = sub_net_list

    def get_output(self, *args, **kwargs):
        return self._get_output(*args, **kwargs)

    def load_param(self, device, pretrain_path):
        for sub_net in self.sub_net_list:
            sub_net.load_param(device, pretrain_path)

    def save_param(self, save_dir, it):
        for sub_net in self.sub_net_list:
            sub_net.save_param(save_dir, it)