%matplotlib inline
import numpy as np
import cv2 as cv
import math

from PIL import Image
import matplotlib.pyplot as plt

#
# Problem set 0: Images as Functions
#


def displayImages(images, title=None, subtitles=[], nbCols=2):
    '''
    Display a list of images, with an optionnal global title and
    optionnal subtitles
    '''
    nbRows = math.ceil(len(images) / nbCols)
    fig, axes = plt.subplots(nbRows, nbCols, squeeze=False);
    if title != None:
        fig.suptitle(title, fontsize=14)

    for imNo, im in enumerate(images):
        col = imNo % nbCols
        row = imNo // nbCols
        axes[row, col].set_aspect('auto')
        if imNo < len(subtitles):
            axes[row, col].set_title(subtitles[imNo])
        # Note: openCV is BGR, matplotlib is RGB
        if len(im.shape) == 3 and im.shape[2] >= 3:
            axes[row, col].imshow(im[:,:,(2, 1, 0)])
        elif len(im.shape) == 2:
            axes[row, col].imshow(im, cmap='gray')
        else:
            print('Not RGB, aRGB or grayscale')
        axes[row, col].tick_params(axis='both', which='both',
        bottom='off', top='off', left='off', right='off',
        labelbottom='off', labeltop='off', labelleft='off', labelright='off');

# %%

im1, im2 = cv.imread('input/ps0-1-a-1.png'), cv.imread('input/ps0-1-a-2.png')
displayImages([im1, im2], 'Input images', ['Image 1', 'Image 2'])

# %%

displayImages([im1[:,:,(2,1,0)], im1[:,:,(0,2,1)], im1[:,:,(1,0,2)],
               im2[:,:,(2,1,0)], im2[:,:,(0,2,1)], im2[:,:,(1,0,2)]],
              title='Color planes swapping',
              subtitles=['B-R', 'G-R', 'B-G'],
              nbCols=3)

# %%

displayImages([im1[:,:,0], im1[:,:,1], im1[:,:,2],
               im2[:,:,0], im2[:,:,1], im2[:,:,2]],
              title='Monochromes',
              subtitles=['Blue', 'Green', 'Red'],
              nbCols=3)

#%%

s = 100
y1, x1 = (im1.shape[0] - s) // 2, (im1.shape[1] - s) // 2
y2, x2 = (im2.shape[0] - s) // 2, (im2.shape[1] - s) // 2

patchwork = im2.copy()
patchwork[y2:y2+s, x2:x2+s, :] = im1[y1:y1+s, x1:x1+s, :]

displayImages([patchwork], title="Patchwork", nbCols=1)

#%%

def printStats(img):
    for i, color in enumerate(['B', 'G', 'R']):
        arr = img[:,:,i]
        print('{}: {}-{} mean {}, std {}'.format(
            color, arr.min(), arr.max(), arr.mean(), arr.std()))

print('Image 1:')
printStats(im1)
print('Image 2:')
printStats(im2)

#%%

dx = -2

transMat = np.float32([[1, 0, dx], [0, 1, 0]])
shifted1 = cv.warpAffine(im1, transMat, (im1.shape[1], im1.shape[0]), borderMode=cv.BORDER_REFLECT_101)
shifted2 = cv.warpAffine(im2, transMat, (im2.shape[1], im2.shape[0]), borderMode=cv.BORDER_REFLECT_101)

displayImages([shifted1, shifted2], title='Shifted {} pixel lefts'.format(dx))

#%%

edges1 = cv.subtract(im1[:,:,1], shifted1[:,:,1])
edges2 = cv.subtract(im2[:,:,1], shifted2[:,:,1])

displayImages([edges1, edges2],
              title='Vertical green edges (shifted substracted from original)')

#%%

def printNoisy(img):
    noisyImgs = []
    subtitles = []
    scales = [1, 5, 10, 20, 40]
    for i, color in enumerate(['B', 'G', 'R']):
        for scale in scales:
            noise = np.random.normal(scale=scale, size=img.shape)
            noisyImg = img.copy()
            noisyImg[:,:,i] = cv.add(noisyImg[:,:,i], noise[:,:,i], dtype=cv.IMREAD_GRAYSCALE)
            noisyImgs.append(noisyImg)
            subtitles.append('{}{}'.format(color, scale))

    displayImages(noisyImgs, subtitles=subtitles, title='Noisy images', nbCols=len(scales))

printNoisy(im1)

#%%

printNoisy(im2)
