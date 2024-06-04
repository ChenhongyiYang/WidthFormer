"""
Author: Chenhongyi Yang
Reference: We are sorry that we cannot find this script's original authors, but we are appreciate about their work.
"""

import glob
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple

import cv2
import lmdb


_10TB = 10 * (1 << 40)

class LmdbDataExporter(object):
    """
    making LMDB database
    """
    label_pattern = re.compile(r'/.*/.*?(\d+)$')

    def __init__(self,
                 img_dirs=None,
                 output_path=None,
                 batch_size=100):
        """
            img_dir: imgs directory
            output_path: LMDB output path
        """
        self.img_dirs = img_dirs
        self.output_path = output_path
        self.batch_size = batch_size
        self.label_list = list()

        for img_dir in img_dirs:
            if not os.path.exists(img_dir):
                raise Exception(f'{img_dir} is not exists!')

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        self.lmdb_env = lmdb.open(output_path, map_size=_10TB, max_dbs=4)
        self.label_dict = defaultdict(int)

    def export(self):
        idx = 0
        results = []
        st = time.time()
        iter_img_lst = self.read_imgs()
        length = self.get_length()
        while True:
            items = []
            try:
                while len(items) < self.batch_size:
                    items.append(next(iter_img_lst))
            except StopIteration:
                break

            with ThreadPoolExecutor() as executor:
                results.extend(executor.map(self._extract_once, items))

            if len(results) >= self.batch_size:
                self.save_to_lmdb(results)
                idx += self.batch_size
                et = time.time()
                print(f'time: {(et-st)}(s)  count: {idx}')
                st = time.time()
                if length - idx <= self.batch_size:
                    self.batch_size = 1
                del results[:]

        et = time.time()
        print(f'time: {(et-st)}(s)  count: {idx}')
        self.save_to_lmdb(results)
        self.save_total(idx)
        print('Total length:', len(results))
        del results[:]

    def save_to_lmdb(self, results):
        """
        persist to lmdb
        """
        with self.lmdb_env.begin(write=True) as txn:
            while results:
                img_key, img_byte = results.pop()
                if img_key is None or img_byte is None:
                    continue
                txn.put(img_key, img_byte)

    def save_total(self, total: int):
        """
        persist all numbers of imgs
        """
        with self.lmdb_env.begin(write=True, buffers=True) as txn:
            txn.put('total'.encode(), str(total).encode())

    def _extract_once(self, item) -> Tuple[bytes, bytes]:
        full_path = item[-1]
        imageKey = item[1]

        img = cv2.imread(full_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img is None:
            print(f'{full_path} is a bad img file.')
            return None, None
        _, img_byte = cv2.imencode('.jpg', img)
        return (imageKey.encode('ascii'), img_byte.tobytes())

    def get_length(self):
        _img_lists = [glob.glob(os.path.join(x, '*.jpg')) for x in self.img_dirs]
        img_list = []
        for l in _img_lists:
            for x in l:
                img_list.append(x)
        return len(img_list)

    def read_imgs(self):
        _img_lists = [glob.glob(os.path.join(x, '*.jpg')) for x in self.img_dirs]
        img_list = []
        for l in _img_lists:
            for x in l:
                img_list.append(x)

        for idx, item_img in enumerate(img_list):
            write_key = os.path.split(item_img)[-1]
            item = (idx, write_key, item_img)
            yield item


if __name__ == '__main__':
    # img_lists = [
    #     '/home/s2139448/projects/bevdet/data/nuscenes/samples/CAM_BACK',
    #     '/home/s2139448/projects/bevdet/data/nuscenes/samples/CAM_BACK_LEFT',
    #     '/home/s2139448/projects/bevdet/data/nuscenes/samples/CAM_BACK_RIGHT',
    #     '/home/s2139448/projects/bevdet/data/nuscenes/samples/CAM_FRONT',
    #     '/home/s2139448/projects/bevdet/data/nuscenes/samples/CAM_FRONT_LEFT',
    #     '/home/s2139448/projects/bevdet/data/nuscenes/samples/CAM_FRONT_RIGHT',
    #
    #     '/home/s2139448/projects/bevdet/data/nuscenes/sweeps/CAM_BACK',
    #     '/home/s2139448/projects/bevdet/data/nuscenes/sweeps/CAM_BACK_LEFT',
    #     '/home/s2139448/projects/bevdet/data/nuscenes/sweeps/CAM_BACK_RIGHT',
    #     '/home/s2139448/projects/bevdet/data/nuscenes/sweeps/CAM_FRONT',
    #     '/home/s2139448/projects/bevdet/data/nuscenes/sweeps/CAM_FRONT_LEFT',
    #     '/home/s2139448/projects/bevdet/data/nuscenes/sweeps/CAM_FRONT_RIGHT',
    # ]
    # output_path = '/scratch_fast/datasets/nuScenes/lmdb/cam_images'

    img_lists = [
        '/cluster_public_data/nuScenes/origin/samples/CAM_BACK',
        '/cluster_public_data/nuScenes/origin/samples/CAM_BACK_LEFT',
        '/cluster_public_data/nuScenes/origin/samples/CAM_BACK_RIGHT',
        '/cluster_public_data/nuScenes/origin/samples/CAM_FRONT',
        '/cluster_public_data/nuScenes/origin/samples/CAM_FRONT_LEFT',
        '/cluster_public_data/nuScenes/origin/samples/CAM_FRONT_RIGHT',

        '/cluster_public_data/nuScenes/origin/sweeps/CAM_BACK',
        '/cluster_public_data/nuScenes/origin/sweeps/CAM_BACK_LEFT',
        '/cluster_public_data/nuScenes/origin/sweeps/CAM_BACK_RIGHT',
        '/cluster_public_data/nuScenes/origin/sweeps/CAM_FRONT',
        '/cluster_public_data/nuScenes/origin/sweeps/CAM_FRONT_LEFT',
        '/cluster_public_data/nuScenes/origin/sweeps/CAM_FRONT_RIGHT',
    ]
    output_path = '/cluster_home/custom_data/datasets/nuscenes/lmdb/cam_images_new'

    exporter_train = LmdbDataExporter(img_lists, output_path, batch_size=10000)
    exporter_train.export()
