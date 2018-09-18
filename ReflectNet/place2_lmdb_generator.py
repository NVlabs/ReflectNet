#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
Authors: Patrick Wieschollek, Orazio Gallo, Jinwei Gu, and Jan Kautz
"""


import tarfile
import argparse
import cv2
import numpy as np
from tensorpack import *

"""
Just convert Place2-dataset (http://places2.csail.mit.edu) to an LMDB file for
more efficient reading.


example:

    python place2_lmdb_generator.py --tar train_large_places365standard.tar \
                                    --lmdb /data/train_large_places365standard.lmdb

    or

    python place2_lmdb_generator.py --tar train_large_places365standard.tar \
                                    --debug
"""


class TarReader(RNGDataFlow):
    """Read images directly from tar file without unpacking
    """
    def __init__(self, tar):
        super(TarReader, self).__init__()
        self.tar = tarfile.open(tar)

    def get_data(self):
        for member in self.tar:
            f = self.tar.extractfile(member)
            jpeg = np.asarray(bytearray(f.read()), dtype=np.uint8)
            f.close()
            yield [jpeg]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tar', help='path to tar file',
                        default='/data/train_large_places365standard.tar')
    parser.add_argument('--lmdb', help='path to database (to be written)',
                        default='/data/train_large_places365standard.lmdb')
    parser.add_argument('--debug', action='store_true',
                        help='just show the images')

    args = parser.parse_args()

    if args.debug:
        ds = TarReader(args.tar)
        ds.reset_state()
        for jpeg in ds.get_data():
            rgb = cv2.imdecode(np.asarray(jpeg), cv2.IMREAD_COLOR)
            cv2.imshow("RGB image from Place2-dataset", rgb)
            cv2.waitKey(0)
    else:
        ds = TarReader(args.tar)
        ds.reset_state()
        dftools.dump_dataflow_to_lmdb(ds, args.lmdb)
