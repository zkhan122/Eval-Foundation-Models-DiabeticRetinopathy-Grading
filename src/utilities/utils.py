import matplotlib.pyplot as plt
import numpy as np

def show_img(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(np.img, (1, 2, 0)))
    plt.show()