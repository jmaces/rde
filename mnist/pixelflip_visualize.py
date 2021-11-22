import os

from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib as tpl


# distortion plot
fig = plt.figure(figsize=(16, 9))
for file in glob(os.path.join("results", "pixelflip-*.npz")):
    data = np.load(file)
    davg = np.mean(data["distortions"], axis=0)
    dstd = np.sqrt(np.mean(np.square(data["deviations"]), axis=0))
    (line,) = plt.plot(
        np.linspace(0, 100, davg.size),
        np.flip(davg),
        label=data["name"],
        linewidth=2,
    )
    plt.fill_between(
        np.linspace(0, 100, davg.size),
        np.flip(davg - dstd),
        np.flip(davg + dstd),
        facecolor=line.get_color(),
        alpha=0.5,
    )
plt.xlabel("rate (percentage of non-randomised components)")
plt.ylabel("distortion (squared distance)")
plt.legend()
plt.autoscale(enable=True, axis="both", tight=True)
plt.tight_layout()

fig.savefig(
    os.path.join("results", "rate-distortion-std.png"),
    format="png",
    bbox_inches="tight",
    dpi=300,
)
tpl.save(
    os.path.join("results", "rate-distortion-std.tex"),
    figureheight="\\figureheight",
    figurewidth="\\figurewidth",
)

# accuracy plot
fig = plt.figure(figsize=(16, 9))
for file in glob(os.path.join("results", "pixelflip-*.npz")):
    data = np.load(file)
    aavg = np.mean(data["accuracies"], axis=0)
    astd = np.sqrt(np.mean(np.square(data["accdeviations"]), axis=0))
    (line,) = plt.plot(
        np.linspace(0, 100, aavg.size),
        np.flip(aavg),
        label=data["name"],
        linewidth=2,
    )
    plt.fill_between(
        np.linspace(0, 100, aavg.size),
        np.flip(aavg - astd),
        np.flip(aavg + astd),
        facecolor=line.get_color(),
        alpha=0.5,
    )
plt.xlabel("rate (percentage of non-randomised components)")
plt.ylabel("accuracy")
plt.legend()
plt.autoscale(enable=True, axis="both", tight=True)
plt.tight_layout()

fig.savefig(
    os.path.join("results", "rate-accuracy-std.png"),
    format="png",
    bbox_inches="tight",
    dpi=300,
)
tpl.save(
    os.path.join("results", "rate-accuracy-std.tex"),
    figureheight="\\figureheight",
    figurewidth="\\figurewidth",
)
