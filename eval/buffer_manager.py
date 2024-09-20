import torch


class BaseBuffer:
    def __init__(self, max_len):
        self.buffer = None
        self.max_len = max_len


    def push(self, new_data):
        if self.buffer is None:
            self.buffer = new_data
        else:
            assert new_data.shape[1:] == self.buffer.shape[1:]
            self.buffer = torch.cat((self.buffer, new_data), dim=0)   # [T, ...]

        if self.__len__() > self.max_len:
            self.buffer = self.buffer[-self.max_len:]

    def __len__(self):
        if self.buffer is None:
            return 0
        else:
            return self.buffer.shape[0]


class FeatureBuffer(BaseBuffer):
    pass


class TrajectoryBuffer(BaseBuffer):
    def push(self, new_data):
        if self.buffer is None or new_data is None:
            self.buffer = new_data
        else:
            assert new_data.shape[1:] == self.buffer.shape[1:]
            self.buffer = torch.cat((self.buffer, new_data), dim=0)  # [T, ...]

        if self.__len__() > self.max_len:
            self.buffer = self.buffer[-self.max_len:]


class TimestampBuffer(BaseBuffer):
    def push(self, new_data):
        if self.buffer is None:
            self.buffer = new_data
        else:
            assert new_data.shape[1:] == self.buffer.shape[1:]
            assert torch.all(new_data[0] >= self.buffer[-1])
            self.buffer = torch.cat((self.buffer, new_data), dim=0)  # [T, ...]

        if self.__len__() > self.max_len:
            self.buffer = self.buffer[-self.max_len:]





