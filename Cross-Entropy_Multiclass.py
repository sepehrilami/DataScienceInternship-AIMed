import numpy as np


def multi_class_cross_entropy_loss(predictions, labels):
    # Calculate multi-class cross entropy loss for every pixel in an image, for every image in a batch.
    # - the first sum is over all classes,
    # - the second sum is over all rows of the image,
    # - the third sum is over all columns of the image
    # - the last mean is over the batch of images.

    # predictions: Output prediction of the neural network.
    # labels: Correct labels.
    # return: Computed multi-class cross entropy loss.

    loss = -np.mean(np.sum(np.sum(np.sum(labels * np.log(predictions), axis=1), axis=1), axis=1))

    return loss
