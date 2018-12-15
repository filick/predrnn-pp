import numpy as np
import random
from metpy.io import Level3File


class InputHandle(object):

    def __init__(self, path, seq_length, batch_size, is_test, test_year=2017):
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.is_test = is_test

        self.meta = []
        test_year_str = str(test_year)
        for line in open(path, 'r'):
            file_id, time_str, file_path = line.strip().split(',')
            if is_test and time_str[:4] != test_year_str:
                continue
            if (not is_test) and time_str[:4] == test_year_str:
                continue
            file_id = int(file_id)
            self.meta.append((file_id, file_path))

        self.begin_idx = []
        i = 0
        while i + seq_length - 1 < len(self.meta):
            if self.meta[i+seq_length-1][0] - self.meta[i][0] == seq_length - 1:
                self.begin_idx.append(i)
                i += 1
            else:
                i = i + seq_length - 1


    def begin(self, do_shuffle = True):
        self.shuffle = do_shuffle
        if self.shuffle:
            self.gen_batch = self._gen_train_batch()
        else:
            self.gen_batch = self._gen_test_batch()


    def next(self):
        pass


    def get_batch(self):
        out = np.zeros((self.batch_size, self.seq_length, 502, 502, 1), "float32")
        for bi, b in enumerate(self.batch_idx):
            for s in range(b, b+self.seq_length):
                data_seq = np.array(Level3File(self.meta[b][1]).sym_block[0][0]['data'])
                out[bi, :, :, :, 0] = data_seq
        return out


    def no_batch_left(self):
        try:
            self.batch_idx = next(self.gen_batch)
            return False
        except StopIteration:
            return True


    def _gen_train_batch(self):
        for _ in range(len(self.meta) // (self.batch_size * self.seq_length) ):
            yield np.random.choice(self.begin_idx, self.batch_size, True)


    def _gen_test_batch(self):
        batch_idx = []
        last = -10000
        for i in self.begin_idx:
            if self.meta[i][0] - last >= self.seq_length:
                batch_idx.append(i)
                last = self.meta[i][0]
            if len(batch_idx) == self.batch_size:
                yield batch_idx
                batch_idx = []