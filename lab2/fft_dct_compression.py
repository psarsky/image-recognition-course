from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

PARENT_DIR = Path(__file__).resolve().parent
IMG_PATH = PARENT_DIR / "lena.jpg"
RESULTS_DIR = PARENT_DIR / "results"
STEP = 20
NOISE_SIGMA = 20
METHOD_NAMES = {
    "f": "FFT",
    "d": "DCT",
}


def add_noise(img):
    return np.clip(img + np.random.normal(0, NOISE_SIGMA, img.shape), 0, 255)


def transform(method, img):
    if method == "f":
        return np.fft.fft2(img, norm="ortho")
    return cv2.dct(img)


def inverse(method, X):
    if method == "f":
        return np.real(np.fft.ifft2(X, norm="ortho"))
    return cv2.idct(X)


def quantize(X):
    if np.iscomplexobj(X):
        return quantize(X.real) + 1j * quantize(X.imag)
    return np.round(X / STEP) * STEP


def spectrum(method, X):
    if method == "f":
        X = np.fft.fftshift(X)
    return np.log1p(np.abs(X))


def psnr(a, b):
    return 10 * np.log10(255**2 / np.mean((a - b) ** 2))


def main():
    img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Cannot load image:", IMG_PATH)
        return
    img = img[: img.shape[0] // 2 * 2, : img.shape[1] // 2 * 2].astype(np.float64)

    while True:
        method = input("Method (f - FFT, d - DCT): ")
        if method in {"f", "d"}:
            break
        print("Incorrect input")

    images = {"original": img, "noisy": add_noise(img)}

    plt.figure(figsize=(12, 9))
    for i, (name, x) in enumerate(images.items()):
        Xq = quantize(transform(method, x))
        rec = np.clip(inverse(method, Xq), 0, 255)
        nonzero = np.count_nonzero(Xq) / Xq.size * 100
        p_in = psnr(img, x) if name == "noisy" else float("inf")
        p_out = psnr(img, rec)
        print(
            f"{name}: non-zero coefficients {nonzero:.2f}%, "
            f"PSNR input {p_in:.2f} dB, PSNR reconstruction {p_out:.2f} dB"
        )

        plt.subplot(2, 3, 3 * i + 1)
        plt.imshow(x, cmap="gray", vmin=0, vmax=255)
        plt.title(f"{name}\nPSNR={p_in:.2f} dB")
        plt.axis("off")

        plt.subplot(2, 3, 3 * i + 2)
        plt.imshow(spectrum(method, Xq), cmap="gray")
        plt.title(f"{METHOD_NAMES[method]} after quantization\nnon-zero={nonzero:.2f}%")
        plt.axis("off")

        plt.subplot(2, 3, 3 * i + 3)
        plt.imshow(rec, cmap="gray", vmin=0, vmax=255)
        plt.title(f"reconstruction\nPSNR={p_out:.2f} dB")
        plt.axis("off")

    plt.suptitle(f"{METHOD_NAMES[method]}, quantization step = {STEP}")
    plt.tight_layout()
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
