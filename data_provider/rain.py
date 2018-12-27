import numpy as np
import os
from .radar import _img_arg

class InputHandle(object):
    rain_data = None
    radar_data = None

    def __init__(self, rain_meta, radar_meta, radar_length, 
                 batch_size, img_width, is_test, test_year=2017,
                 threshold=350):
        self.radar_length = radar_length
        self.is_test = is_test
        self.batch_size = batch_size
        self.img_width = img_width

        process_radar_line = lambda line: int(line.split(',')[0])
        self.radar_meta = np.array([process_radar_line(line) for line in open(radar_meta, 'r')], dtype='int')

        def process_rain_line(line):
            items = line.split(',')
            return [int(items[2]), int(items[3]), int(items[0][:4])]
        self.rain_meta = np.array([process_rain_line(line) for line in open(rain_meta, 'r')], dtype='int')

        self.rain_valid = np.ones(shape=(self.rain_meta.shape[0],), dtype='int') 
        if InputHandle.rain_data is None:
            dirname = os.path.dirname(rain_meta)
            filename = os.path.basename(rain_meta)
            InputHandle.rain_data = np.load(os.path.join(dirname, filename.split('-')[-1] + '.npy'))
            high = InputHandle.rain_data.max(axis=(1,2))
            low  = InputHandle.rain_data.min(axis=(1,2))
            self.rain_valid[(self.rain_data > (0.6*high + 0.4*low).reshape((-1,1,1))).sum(axis=(1,2)) < 10] = 0
            self.rain_valid[(high - low) < 1e-3] = 0
            self.rain_valid[high > 450] = -1
            self.rain_valid[low < -1] = -1
 
            InputHandle.rain_data[InputHandle.rain_data > 450] = 450
            InputHandle.rain_data[InputHandle.rain_data < 1e-3] = 0
            InputHandle.rain_data[np.where(self.rain_valid==0)[0], :, :] = 0

        if InputHandle.radar_data is None:
            dirname = os.path.dirname(radar_meta)
            filename = os.path.basename(radar_meta)
            InputHandle.radar_data = np.load(os.path.join(dirname, filename.split('-')[-1] + '.npy'))

        self.selected_rain_idx = []
        self.selected_radar_idx = []
        i = radar_length - 1
        j = 0

        while i < self.radar_meta.shape[0] and j < self.rain_meta.shape[0]:
            y = self.rain_meta[j, 2]
            if is_test and y != test_year:
                j += 1
            elif (not is_test) and y == test_year:
                j += 1
            elif self.rain_valid[j] == -1:
                j += 1
            elif self.rain_valid[j] == 0 and np.random.rand() > 0.1:
                j += 1
            elif abs(self.rain_meta[j, 1]) > threshold:
                j += 1
            elif (self.radar_meta[i] - self.radar_meta[i - radar_length + 1]) > radar_length:
                i += 1
            elif self.radar_meta[i] < self.rain_meta[j, 0]:
                i += 1
            elif self.radar_meta[i] > self.rain_meta[j, 0]:
                j += 1
            else:
                self.selected_radar_idx.append(i - radar_length + 1)
                self.selected_rain_idx.append(j)
                j += 1

        assert len(self.selected_radar_idx) == len(self.selected_rain_idx)

    def begin(self, do_shuffle = True):
        self.shuffle = do_shuffle
        if self.shuffle:
            self.gen_batch = self._gen_train_batch()
        else:
            self.gen_batch = self._gen_test_batch()


    def next(self):
        pass


    def get_batch(self):
        out = np.zeros((len(self.batch_idx), self.radar_length + 1, self.img_width, self.img_width, 1), "float32")
        for bi, idx in enumerate(self.batch_idx):
            i, j = self.selected_radar_idx[idx], self.selected_rain_idx[idx]
            for s in range(i, i+self.radar_length):
                img = np.array(InputHandle.radar_data[s], dtype='float32')
                img = _img_arg(img, self.img_width, None)
                out[bi, s-i, :, :, 0] = img
            img = np.array(InputHandle.rain_data[j], dtype='float32')
            img = _img_arg(img, self.img_width, None)
            out[bi, -1, :, :, 0] = img
        return out


    def no_batch_left(self):
        try:
            self.batch_idx = next(self.gen_batch)
            return False
        except StopIteration:
            return True


    def _gen_train_batch(self):
        choice = list(range(len(self.selected_radar_idx)))
        for _ in range(len(choice) // self.batch_size):
            yield np.random.choice(choice, self.batch_size, True)


    def _gen_test_batch(self):
        i = 0
        while i + self.batch_size <= len(self.selected_radar_idx):
            yield list(range(i, i+self.batch_size))
            i += self.batch_size
