import json
import pykrige.kriging_tools as kt
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging
import numpy as np


def interpolate(jsonfile, grid_lon, grid_lat):
    min_lon = min(grid_lon[0], grid_lon[-1]) - 0.1 * \
        abs(grid_lon[0] - grid_lon[-1])
    max_lon = max(grid_lon[0], grid_lon[-1]) + 0.1 * \
        abs(grid_lon[0] - grid_lon[-1])
    min_lat = min(grid_lat[0], grid_lat[-1]) - 0.1 * \
        abs(grid_lat[0] - grid_lat[-1])
    max_lat = max(grid_lat[0], grid_lat[-1]) + 0.1 * \
        abs(grid_lat[0] - grid_lat[-1])

    obj = json.load(open(jsonfile, 'r'))

    def parse(item):
        return [float(item['Lat']), float(item['Lon']), float(item['PRE_1h'])]

    observed = np.array(list(map(parse, obj['DS'])))
    observed = observed[observed[:, 0] > min_lat]
    observed = observed[observed[:, 0] < max_lat]
    observed = observed[observed[:, 1] > min_lon]
    observed = observed[observed[:, 1] < max_lon]
    observed = observed[observed[:, 2] < 10000]
    interp_model = UniversalKriging(observed[:, 0], observed[:, 1], observed[:, 2], variogram_model='linear',
                      drift_terms=['regional_linear'])
                                   
    z, ss = interp_model.execute('grid', grid_lat, grid_lon)
    import IPython
    IPython.embed()


if __name__ == '__main__':
    grid_lon = np.arange(81.2547209531, 86.95, 0.09038990625)
    grid_lat = np.arange(42.3490884375, 46.447, 0.064536875)
    interpolate(
        '/Users/filick/Documents/work/mini/shikaifang/2018/01/01/rgwst.hour.201801010000_UTC', grid_lon, grid_lat)
