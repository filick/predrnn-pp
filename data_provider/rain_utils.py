import json
import os
import datetime
from pykrige.ok import OrdinaryKriging
import numpy as np


class _RadarMetaHelper(object):

    def __init__(self, path):
        self.file = open(path, 'r')
        self.readed_next = False
        self.cur_line = self._parse_line(self.file.readline())
        self.next_line = None

    def _parse_line(self, line):
        items = line.split(',')
        return (int(items[0]), items[1])

    def move(self):
        if self.has_next():
            self.cur_line = self.next_line
            self.next_line = None
            self.readed_next = False

    def get_current(self):
        return self.cur_line

    def get_next(self):
        self.has_next()
        return self.next_line

    def has_next(self):
        if self.readed_next:
            return bool(self.next_line)
        else:
            self.readed_next = True
            try:
                self.next_line = self._parse_line(self.file.readline())
            except IOError:
                self.next_line = None
                return False
            return True

    def __del__(self):
        try:
            self.file.close()
        except:
            pass


def process(rain_file_root, radar_meta, grid_lon, grid_lat):
    min_lon = min(grid_lon[0], grid_lon[-1]) - 0.1 * \
        abs(grid_lon[0] - grid_lon[-1])
    max_lon = max(grid_lon[0], grid_lon[-1]) + 0.1 * \
        abs(grid_lon[0] - grid_lon[-1])
    min_lat = min(grid_lat[0], grid_lat[-1]) - 0.1 * \
        abs(grid_lat[0] - grid_lat[-1])
    max_lat = max(grid_lat[0], grid_lat[-1]) + 0.1 * \
        abs(grid_lat[0] - grid_lat[-1])

    helper = _RadarMetaHelper(radar_meta)
    rain_writer = open(os.path.join(
        rain_file_root, os.path.basename(radar_meta)), 'w')

    block = []
    for y in sorted(os.listdir(rain_file_root)):
        y_path = os.path.join(rain_file_root, y)
        if not os.path.isdir(y_path):
            continue
        for m in sorted(os.listdir(y_path)):
            print('Processing %s-%s' % (y, m))
            m_path = os.path.join(y_path, m)
            for d in sorted(os.listdir(m_path)):
                d_path = os.path.join(m_path, d)
                for f in sorted(os.listdir(d_path)):
                    file_path = os.path.abspath(os.path.join(d_path, f))
                    time_str = f.split('.')[-1].split('_')[0]
                    time = datetime.datetime.strptime(time_str, '%Y%m%d%H%M%S')

                    obj = json.load(open(file_path, 'r'))
                    observed = np.array(list(map(lambda item: [float(item['Lat']), float(
                        item['Lon']), float(item['PRE_1h'])], obj['DS'])))
                    observed = observed[observed[:, 0] > min_lat]
                    observed = observed[observed[:, 0] < max_lat]
                    observed = observed[observed[:, 1] > min_lon]
                    observed = observed[observed[:, 1] < max_lon]
                    observed = observed[observed[:, 2] < 10000]

                    try:
                        interp_model = OrdinaryKriging(observed[:, 0], observed[:, 1], observed[:, 2],
                                                       variogram_model='exponential', verbose=False, enable_plotting=False)
                        z, _ = interp_model.execute('grid', grid_lat, grid_lon)
                    except Exception as e:
                        print('Error with %s' % file_path)
                        print(e)
                        continue

                    block.append(z)
                    max_rain = np.max(observed[:, 2])

                    while helper.has_next() and helper.get_next()[1] < time_str:
                        helper.move()

                    rid, rtime_str = helper.get_current()
                    rtime = datetime.datetime.strptime(
                        rtime_str, '%Y%m%d%H%M%S')
                    rdelta = (time - rtime).total_seconds()

                    if rdelta > 0 and helper.has_next():
                        nid, ntime_str = helper.get_next()
                        ntime = datetime.datetime.strptime(
                            ntime_str, '%Y%m%d%H%M%S')
                        ndelta = (time - ntime).total_seconds()
                        if abs(ndelta) < abs(rdelta):
                            rid, rdelta = nid, ndelta
                            helper.move()

                    rain_writer.write(','.join([time_str, str(max_rain), str(
                        rid), str(int(rdelta)), file_path]) + os.linesep)

    rain_writer.flush()
    rain_writer.close()

    block = np.stack(block)
    outname = os.path.basename(radar_meta)
    np.save(os.path.join(rain_file_root, outname.split('-')[-1] + '.npy'), block)


if __name__ == '__main__':
    grid_lon = np.arange(81.2547209531, 86.95, 0.09038990625)
    grid_lat = np.arange(46.41491156, 42.3490, -0.064536875)
    process('../data/rain', '../data/radar/list-Z9080', grid_lon, grid_lat)
