import numpy as np
from PIL import Image
from math import sqrt
import os


identity = [[0, 0, 0],
           [0, 1, 0],
           [0, 0, 0]]

ed1 = [[1, 0, -1],
       [0, 0, 0],
       [-1, 0, 1]]
ed3 = [[0, 1, 0],
       [1, -4, 1],
       [0, 1, 0]]
ed4 = [[-1, -1, -1],
       [-1, 8, -1],
       [-1, -1, 1]]

sharpen = [[0, -1, 0],
           [-1, 5, -1],
           [0, -1, 0]]

box_blur = [[1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]]

gaus_blur = [[1, 2, 1],
             [2, 4, 1],
             [1, 2, 1]]

gaus_blur2 = [[1, 4, 6, 4, 1],
              [4, 16, 24, 16, 4],
              [6, 24, 36, 24, 6],
              [4, 16, 24, 16, 4],
              [1, 4, 6, 4, 1]]

unsharp_ma = [[1, 4, 6, 4, 1],
              [4, 16, 24, 16, 4],
              [6, 24, -476, 24, 6],
              [4, 16, 24, 16, 4],
              [1, 4, 6, 4, 1]]

kernels = [["identity.jpg", identity, 1], ["edgeDetect1.jpg", ed1, 1], ["edgeDetect2.jpg", ed3, 1],
           ["edgeDetect3.jpg", ed4, 1], ["sharpen.jpg", sharpen, 1], ["boxBlur.jpg", box_blur, 9],
           ["gausBlur3x3.jpg", gaus_blur, 16],["guasBlur5x5.jpg", gaus_blur2, 256],
           ["unsharp.jpg", unsharp_ma, -256]]
           
file_data = np.genfromtxt("channel2.csv", delimiter=",")
results_file = "results"

def convolve(csv, kernel, div):
    x_dim = csv.shape[0]
    y_dim = csv.shape[1]
    new_arr = []
    i = 0
    j = 0
    ker_len = len(kernel[0])
    new_arr = []
    while(True):
        tot = 0
        for k in range(ker_len):
            for l in range(ker_len):
                tot += (kernel[k][l] * csv[k + i][l + j])
        new_arr.append(tot / div)
        j += 1
        if j >= x_dim - ker_len + 1:
            j = 0
            i += 1
            if i >= y_dim - ker_len + 1:
                break
        
    #width = int(sqrt(len(new_arr)))
    return np.asarray(new_arr).reshape(x_dim - ker_len + 1, y_dim - ker_len + 1)

if not os.path.exists("results"):
    try:
        os.mkdir("results")
    except osError:
        print("Creating results directory failed")
        exit()

for k in kernels:
    
    img = Image.fromarray(convolve(convolve(np.copy(file_data), k[1], k[2]), k[1], k[2])).convert("L")
    img.save("{}/{}".format(results_file, k[0]))

