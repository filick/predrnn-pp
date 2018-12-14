import numpy as np
import os
import datetime


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
                for file in filter(lambda name: name.endswith('.bin'), os.listdir(r)):
                    time_str = file.split('_')[4]
                    time = datetime.datetime.strptime(time_str, '%Y%m%d%H%M%S')
                    output.write("%d,%s,%s"%(i, time_str, os.path.join(r, file)))
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


if __name__ == '__main__':
    gen_filelist('../data/radar')