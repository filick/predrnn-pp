import numpy as np
import os
import datetime
from concurrent.futures import ThreadPoolExecutor, wait
from metpy.io import Level3File


def gen_filelist(root):
    threshould = datetime.timedelta(minutes=8)
    std_delta_sec  = 330

    for site in os.listdir(root):
        site_folder = os.path.join(root, site)
        if os.path.isfile(site_folder):
            continue

        output = open(os.path.join(root, 'list-%s' % site), 'w')
        i = 0
        cur_time = 0
        flush_count = 0

        def sorted_walk(r):
            nonlocal i, cur_time, flush_count
            if r.endswith(os.sep + 'CR'):
                for file in filter(lambda name: name.endswith('.bin'), sorted(os.listdir(r))):
                    time_str = file.split('_')[4]
                    time = datetime.datetime.strptime(time_str, '%Y%m%d%H%M%S')
                    output.write("%d,%s,%s"%(i, time_str, os.path.realpath(os.path.join(r, file))))
                    output.write("\n")
                    if i == 0 or time - cur_time < threshould:
                        i += 1
                    else:
                        i += max(round((time - cur_time).total_seconds() / std_delta_sec), 1)
                    cur_time = time

                    flush_count += 1
                    if flush_count % 200 == 0:
                        output.flush()
            else:
                for item in sorted(os.listdir(r)):
                    nr = os.path.join(r, item)
                    if os.path.isdir(nr):
                        sorted_walk(nr)

        sorted_walk(root)

        output.flush()
        output.close()


def convert_to_npy(filelist, workers=10):
    filepaths = [line.strip().split(',')[-1] for line in open(filelist, 'r')]
    block = np.zeros((len(filepaths), 502, 502), dtype='unit8')

    def task(tast_id):
        nonlocal block, workers, filepaths
        for i in range(tast_id, len(block), workers):
            img = np.array(Level3File(filepaths[i]).sym_block[0][0]['data'], dtype='uint8')
            block[i,:] = img

    task_pool = ThreadPoolExecutor(max_workers=workers)
    tasks = [task_pool.submit(task, i) for i in range(workers)]
    wait(tasks)
    np.save(block, filelist.split('-')[-1] + '.npz')


if __name__ == '__main__':
    # gen_filelist('../data/radar')
    convert_to_npy('../data/radar/list-Z9080')