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
from mmcv import FileClient
import numpy as np

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

        self.file_client = FileClient(backend='disk')



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
        pointKey = item[1]

        try:
            pts_bytes = self.file_client.get(full_path)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except:
            print(f'{full_path} is a bad img file.')
            return None, None
        return (pointKey.encode('ascii'), points.tobytes())

    def get_length(self):
        _img_lists = [glob.glob(os.path.join(x, '*.bin')) for x in self.img_dirs]
        img_list = []
        for l in _img_lists:
            for x in l:
                img_list.append(x)
        return len(img_list)

    def read_imgs(self):
        _img_lists = [glob.glob(os.path.join(x, '*.bin')) for x in self.img_dirs]
        img_list = []
        for l in _img_lists:
            for x in l:
                img_list.append(x)

        for idx, item_img in enumerate(img_list):
            write_key = os.path.split(item_img)[-1]
            item = (idx, write_key, item_img)
            yield item


if __name__ == '__main__':
    img_lists = [
        '/home/s2139448/projects/bevdet/data/nuscenes/samples/LIDAR_TOP',
        '/home/s2139448/projects/bevdet_new/data/nuscenes/sweeps/LIDAR_TOP'
    ]
    output_path = '/scratch_fast/datasets/nuScenes/lmdb/lidar'
    #
    # img_lists = [
    #     '/cluster_public_data/nuScenes/origin/samples/LIDAR_TOP',
    # ]
    # output_path = '/cluster_home/custom_data/datasets/nuscenes/lmdb/lidar'

    exporter_train = LmdbDataExporter(img_lists, output_path, batch_size=3000)
    exporter_train.export()
