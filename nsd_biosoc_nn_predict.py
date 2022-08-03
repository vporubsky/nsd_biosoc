"""
Developer Name: Veronica Porubsky
Developer ORCID: 0000-0001-7216-3368
Developer GitHub Username: vporubsky
Developer Email: verosky@uw.edu

Description: shows how to use the NSDBioSocC3D network to predict social/non-social from fMRI data.

* This file is under development.

"""
# Import packages
import numpy as np
import torch
from torch.autograd import Variable
from os.path import join
from glob import glob
import skimage.io as io # Scikit-image, used for processing images
from skimage.transform import resize
from NSDBioSocNeuralNetwork import NSDBioSocC3D


def get_fMRI(fMRI_file):
    """
    Loads the fMRI file to be fed to C3D for classification.

    It not clear what format the file will be in, or if it could be processed from a numpy array of voxel BOLD values

    Parameters
    ----------
    fMRI_file: str
        the name of the fMRI_file

    Returns
    -------
    Tensor
        a pytorch batch (n, ch, fr, h, w).
    """
    fMRI = sorted(glob(join('data', fMRI_file, '*.png'))) # Todo: determine input data representation and size
    fMRI = np.array([resize(io.imread(frame), output_shape=(112, 200), preserve_range=True) for frame in clip]) # Todo: check if need to resize

    # Todo: determine if this is required/ what ch, fr, h, and w stand for -->
    # https://stackoverflow.com/questions/67087131/what-is-nchw-format#:~:text=NCHW%20stands%20for%3A,as%20a%201%2DD%20array.
    # https://oneapi-src.github.io/oneDNN/dev_guide_understanding_memory_formats.html
    fMRI = fMRI.transpose(3, 0, 1, 2)  # ch, fr, h, w
    fMRI = np.expand_dims(fMRI, axis=0)  # batch axis
    fMRI = np.float32(fMRI)

    return torch.from_numpy(fMRI)


def main():
    """
    Main function.
    """

    # load a clip to be predicted
    X = get_fMRI('filename')
    X = Variable(X)
    X = X.cuda()

    # get network pretrained model
    net = NSDBioSocC3D()
    net.load_state_dict(torch.load('NSDBioSocC3D.pickle')) # can use to load trained network from nsd_biosoc_nn_train.py
    net.cuda()
    net.eval()

    # perform prediction
    prediction = net(X)
    prediction = prediction.data.cpu().numpy()

    # print prediction
    top_inds = prediction[0].argsort()[::-1][:5]  # reverse sort and take five largest items
    pred_val = prediction[0]
    if pred_val == 0:
        print(f"The subject saw a non-social scene.")
    elif pred_val == 1:
        print(f"The subject saw a social scene.")


# entry point
if __name__ == '__main__':
    main()
