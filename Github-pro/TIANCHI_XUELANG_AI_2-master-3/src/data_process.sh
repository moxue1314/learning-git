#!/usr/bin/env bash
python extract_pic_xml.py &&
python split.10.py &&
python DataAugForTrainAndTest.py &&
python copy_to_train.py