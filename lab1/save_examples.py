from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PARENT_DIR = Path(__file__).resolve().parent


def main():
    names = np.array(["circle", "rectangle", "diamond", "triangle"])
    data = np.load(PARENT_DIR / "database.npz")
    imgs, labels = data["imgs"], data["labels"]

    out_dir = PARENT_DIR / "examples"
    out_dir.mkdir(exist_ok=True)

    for i in np.random.choice(len(imgs), 10, replace=False):
        plt.imshow(imgs[i], cmap="gray")
        plt.title(names[labels[i]])
        plt.savefig(out_dir / f"{i}_{names[labels[i]]}.png", dpi=100)
        plt.close()


if __name__ == "__main__":
    main()
