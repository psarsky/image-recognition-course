from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

PARENT_DIR = Path(__file__).resolve().parent
DB_PATH = PARENT_DIR / "database.npz"
RESULTS_DIR = PARENT_DIR / "results"
METHOD_NAMES = {
    "e": "euclidean distance",
    "c": "cosine distance",
    "k": "k-nearest neighbors",
}


def make_shape(c):
    img = np.zeros((64, 64), np.uint8)
    if c == 0:
        cv2.circle(img, (32, 32), 20, 255, -1)
    elif c == 1:
        cv2.rectangle(img, (14, 20), (50, 44), 255, -1)
    elif c == 2:
        cv2.fillPoly(img, [np.array([[32, 10], [54, 32], [32, 54], [10, 32]])], 255)
    else:
        cv2.fillPoly(img, [np.array([[32, 10], [54, 52], [10, 52]])], 255)
    return img


def transform(img):
    M = cv2.getRotationMatrix2D(
        (32, 32), np.random.uniform(-30, 30), np.random.uniform(0.7, 1.1)
    )
    M[0, 1] += np.random.uniform(-0.2, 0.2)
    M[:, 2] += np.random.uniform(-5, 5, 2)
    return cv2.warpAffine(img, M, (64, 64), flags=cv2.INTER_NEAREST)


def descriptor(img):
    F = np.abs(np.fft.fftshift(np.fft.fft2(img)))
    d = F[28:37, 28:37].flatten()
    return d / d.max()


def match(method, db_desc, db_labels, q):
    if method == "c":
        dist = 1 - db_desc @ q / (np.linalg.norm(db_desc, axis=1) * np.linalg.norm(q))
    else:
        dist = np.linalg.norm(db_desc - q, axis=1)
    best = np.argmin(dist)
    if method == "k":
        nn = np.argsort(dist)[:5]
        pred = np.bincount(db_labels[nn]).argmax()
        best = nn[db_labels[nn] == pred][0]
    return best, dist[best]


def main():
    names = np.array(["circle", "rectangle", "diamond", "triangle"])

    if DB_PATH.exists():
        data = np.load(DB_PATH)
        db_imgs, db_desc, db_labels = data["imgs"], data["desc"], data["labels"]
        print("Loaded database from", DB_PATH)
    else:
        db_labels = np.repeat(np.arange(4), 100)
        db_imgs = np.array([transform(make_shape(c)) for c in db_labels])
        db_desc = np.array([descriptor(x) for x in db_imgs])
        np.savez(DB_PATH, imgs=db_imgs, desc=db_desc, labels=db_labels)
        print("Generated new database:", DB_PATH)

    while True:
        choice = input("Class (0 - circle, 1 - rectangle, 2 - diamond, 3 - triangle): ")
        if choice in {"0", "1", "2", "3"}:
            qc = int(choice)
            break
        print("Incorrect input")

    query = transform(make_shape(qc))
    q = descriptor(query)

    while True:
        method = input("Method (e - euclid, c - cos, k - knn, a - all): ")
        if method in {"e", "c", "k", "a"}:
            break
        print("Incorrect input")

    methods = ["e", "c", "k"] if method == "a" else [method]
    n = len(methods) + 1

    print("query class:", names[qc])
    plt.figure(figsize=(4 * n, 4.5))
    plt.subplot(1, n, 1)
    plt.imshow(query, cmap="gray")
    plt.title("query: " + names[qc])

    for i, m in enumerate(methods):
        best, d = match(m, db_desc, db_labels, q)
        print(f"{METHOD_NAMES[m]}: found {names[db_labels[best]]}, distance {d}")
        plt.subplot(1, n, i + 2)
        plt.imshow(db_imgs[best], cmap="gray")
        plt.title(f"{METHOD_NAMES[m]}\n{names[db_labels[best]]}, d={d:.4f}")

    fig = plt.gcf()
    plt.show()

    if input("Type 'y' or 'Y' to save file: ") in {"y", "Y"}:
        RESULTS_DIR.mkdir(exist_ok=True)
        fig.savefig(
            RESULTS_DIR / f"result_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png",
            dpi=100,
        )


if __name__ == "__main__":
    main()
