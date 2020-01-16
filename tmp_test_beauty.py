import argparse
from beautify import Makeup
import copy
import cv2
import numpy as np
import time


st_time = time.clock()
parser = argparse.ArgumentParser(description="face beautification")
parser.add_argument("--path", type=str, required=True)
args = parser.parse_args()
path = args.path

raw_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
makeup_obj = Makeup(raw_img)
print("Initialization takes {} seconds".format(time.clock() - st_time))

# testing
original = copy.deepcopy(raw_img)
results = makeup_obj.beautify(smooth_val=0.85, whiten_val=0.3)

compa_img = np.concatenate((original, results), axis=1)

cv2.imwrite("beauty_results.ppg", compa_img)

print("Total time usage: {}".format(time.clock() - st_time))
