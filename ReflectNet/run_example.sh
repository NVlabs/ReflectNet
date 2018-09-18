#!/usr/bin/env bash

# Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
# Authors: Patrick Wieschollek, Orazio Gallo, Jinwei Gu, and Jan Kautz

# convert raw to ppm
cd example
dcraw -v -W -g 1 1 -6 *.ARW
cd ..

# run separation
i0=example/DSC01908.ppm
i45=example/DSC01909.ppm
i90=example/DSC01910.ppm
prefix=bar
scale=0.25

python eval.py --scale ${scale} --i0 ${i0} --i45 ${i45} --i90 ${i90} --out example/ --prefix ${prefix}
