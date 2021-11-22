import os

import numpy as np

from tqdm import tqdm

import instances
import models


# PARAMETERS
INDICES = range(0, 1200, 100)  # data samples
NUM_SAMPLES = 16
GROUP_SIZE = 1024


def mapping_to_ordering(mapping, reverse=False, flatten=True):
    """Converts a relevance mapping into a component ordering.

    Arguments:
        mapping: A relevance mapping.
        reverse: Sort in reverse (descending) ordering.
        flatten: Return linear indices into the flattened array.

    Returns:
        The component relevance ordering.

    """
    # break ties randomly by using lexsort instead of simply np.argsort
    random_shift = np.random.randn(*mapping.shape)
    idx = np.lexsort((random_shift.flatten(), mapping.flatten()))
    if reverse:
        idx = idx[::-1]
    if not flatten:
        idx = np.unravel_index(idx, mapping.shape)
    return idx


def distortion_test(
    x, label, mapping, model, batch_size=NUM_SAMPLES, group_size=GROUP_SIZE
):
    """Pixel-flipping based relevance mapping analysis.

    Arguments:
        x: The data sample to analyze.
        label: True data label (one-hot encoded).
        mapping: The relevance mapping to analyze.
        model: A Keras classifier model.
        batch_size: Number of corrupted data copies to sample per step.
        group_size: Size of groups of components to be corrupted at each step.

    Returns:
        Arrays containing the mean distortions, their standard deviations,
        mean accuracies, and their standard deviations for each step
        of the pixel-flipping test.

    """
    xmin, xmax = x.min(), x.max()
    # get predictions and target value
    pred = model.predict(np.expand_dims(x, 0))
    node = np.argmax(pred[0, ...], axis=-1)
    if not node == np.argmax(label):
        print(
            "WARNING: This data sample was misclassified. This can result "
            "in an unexpected behaviour of the distortion test."
        )
    target = pred[0, node]
    # get ordering from mask and number of masking groups
    num_groups = np.ceil(x.size / group_size).astype(np.int)
    ordering = mapping_to_ordering(mapping)
    # initialize empty mask and result arrays
    mask = np.zeros([1] + list(x.shape))
    distortions = np.zeros([num_groups + 1])
    deviations = np.zeros([num_groups + 1])
    accuracies = np.zeros([num_groups + 1])
    accdeviations = np.zeros([num_groups + 1])
    for step in tqdm(range(num_groups + 1), "randomized component groups"):
        # update mask according to component ordering
        mask.flat[ordering[(step - 1) * group_size : step * group_size]] = 1.0
        # simulate corrupted data and predictions
        noise_array = xmin + (xmax - xmin) * np.random.rand(
            batch_size, *x.shape
        )
        corrupted = np.clip(
            (1 - mask) * np.expand_dims(x, 0) + mask * noise_array, xmin, xmax
        )
        corrupted_preds = model.predict(corrupted)
        # compute statistics for step
        distortions[step] = np.mean(
            np.square(corrupted_preds[..., node] - target)
        )
        deviations[step] = np.std(
            np.square(corrupted_preds[..., node] - target)
        ) / np.sqrt(batch_size)
        accuracies[step] = np.mean(
            np.equal(np.argmax(corrupted_preds, axis=-1), np.argmax(label))
        )
        accdeviations[step] = np.std(
            np.equal(np.argmax(corrupted_preds, axis=-1), np.argmax(label))
        ) / np.sqrt(batch_size)
    return (
        distortions,
        deviations,
        accuracies,
        accdeviations,
    )


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

    for file in all_mappings:
        # init result lists
        collected_distortions = []
        collected_deviations = []
        collected_accuracies = []
        collected_accdeviations = []
        # iterate data samples
        for INDEX in INDICES:
            x, y = generator[INDEX][0][0, ...], generator[INDEX][1][0, ...]
            xname = os.path.splitext(
                os.path.split(generator.filenames[INDEX])[1]
            )[0]
            # load mapping
            path = os.path.join("results", xname, file)
            data = np.load(path)
            assert data["index"] == INDEX
            # compute distortion curves
            mapping = data["mapping"]
            results = distortion_test(x, y, mapping, model)
            distortions, deviations, accuracies, accdeviations = results
            # collect results
            collected_distortions.append(distortions)
            collected_deviations.append(deviations)
            collected_accuracies.append(accuracies)
            collected_accdeviations.append(accdeviations)
        # save results
        np.savez_compressed(
            os.path.join("results", "pixelflip-" + file),
            distortions=np.asarray(collected_distortions),
            deviations=np.asarray(collected_deviations),
            accuracies=np.asarray(collected_accuracies),
            accdeviations=np.asarray(collected_accdeviations),
            name=os.path.splitext(file)[0],
        )
