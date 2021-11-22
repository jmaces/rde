import os

import numpy as np

import instances
import models


# PARAMETERS
INDICES = range(0, 1200, 100)  # data samples


def pearson_correlation(x, y):
    """ Pearson correlation coefficient of two flattened arrays. """
    return np.corrcoef(x.flatten(), y.flatten())[0, 1]


def jaccard_index(x, y):
    """ Jaccard index of the binarizations of two arrays. """
    # threshold to binary
    xh = np.heaviside(x - np.mean(x), 0)
    yh = np.heaviside(y - np.mean(y), 0)
    # compute intersection and union
    xb = xh.astype(np.bool)
    yb = yh.astype(np.bool)
    intersection = np.logical_and(xb, yb)
    union = np.logical_or(xb, yb)
    # Jaccard index is the ration intersection over union (IoU)
    return intersection.sum() / union.sum()


if __name__ == "__main__":
    # load model
    model = models.load_model(softmax=True)

    # load instances
    generator, _ = instances.load_generator(class_mode="categorical")

    # collect mappings
    all_mappings = set()
    for root, dirs, files in os.walk("results"):
        all_mappings.update(
            [f for f in files if (f.endswith(".npz") and "pixelflip" not in f)]
        )
    print("Found data for {} mapping methods.".format(len(all_mappings)))

    names = []
    pearson_means = []
    jaccard_means = []
    for file in all_mappings:
        # init result lists
        collected_pearson = []  # for Pearson correlation coefficients
        collected_jaccard = []  # for Jaccard indeces (intersection over union)
        # iterate data samples
        for INDEX in INDICES:
            x, y = generator[INDEX][0][0, ...], generator[INDEX][1][0, ...]
            xname = os.path.splitext(
                os.path.split(generator.filenames[INDEX])[1]
            )[0]
            # load mapping
            path = os.path.join("results", xname, file)
            data = np.load(path)
            mask_path = os.path.join("results", xname, "mask.npz")
            mask_data = np.load(mask_path)
            mask = mask_data["mapping"]
            mapping = data["mapping"]
            assert mask_data["index"] == INDEX
            assert data["index"] == INDEX
            # compute similarity measures
            pearson = pearson_correlation(mapping, mask)
            jaccard = jaccard_index(mapping, mask)
            # collect results
            collected_pearson.append(pearson)
            collected_jaccard.append(jaccard)
        if "mask" not in file:
            names.append(os.path.splitext(file)[0])
            pearson_means.append(np.asarray(collected_pearson).mean())
            jaccard_means.append(np.asarray(collected_jaccard).mean())
        print(os.path.splitext(file)[0])
        print(
            "\tPearson:\t{:1.3e} +- {:1.3e}".format(
                np.asarray(collected_pearson).mean(),
                np.asarray(collected_pearson).std(),
            )
        )
        print(
            "\tJaccard:\t{:1.3e} +- {:1.3e}".format(
                np.asarray(collected_jaccard).mean(),
                np.asarray(collected_jaccard).std(),
            )
        )
        # save results
        np.savez_compressed(
            os.path.join("results", "mask-eval-" + file),
            pearson_correlation=np.asarray(collected_pearson),
            jaccard_index=np.asarray(collected_jaccard),
            name=os.path.splitext(file)[0],
        )
