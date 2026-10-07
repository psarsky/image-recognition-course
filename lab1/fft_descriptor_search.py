import cv2
import matplotlib.pyplot as plt
import numpy as np


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
    M = cv2.getRotationMatrix2D((32, 32), np.random.uniform(-30, 30), np.random.uniform(0.7, 1.1))
    M[0, 1] += np.random.uniform(-0.2, 0.2)
    M[:, 2] += np.random.uniform(-5, 5, 2)
    return cv2.warpAffine(img, M, (64, 64), flags=cv2.INTER_NEAREST)

def descriptor(img):
    F = np.abs(np.fft.fftshift(np.fft.fft2(img)))
    d = F[28:36, 28:36].flatten()
    return d / d.max()

names = np.array(["circle", "rectangle", "rhombus", "triangle"])
db_labels = np.repeat(np.arange(4), 20)
db_imgs = [transform(make_shape(c)) for c in db_labels]
db_desc = np.array([descriptor(x) for x in db_imgs])
np.save("descriptors.npy", db_desc)

qc = np.random.randint(4)
query = transform(make_shape(qc))
q = descriptor(query)

while True:
    method = input("Method (euclid/cos/knn): ")
    if method in {"euclid", "cos", "knn"}:
        break
    print("Incorrect input, valid methods: euclid, cos, knn")

if method == "cos":
    dist = 1 - db_desc @ q / (np.linalg.norm(db_desc, axis=1) * np.linalg.norm(q))
else:
    dist = np.linalg.norm(db_desc - q, axis=1)
best = np.argmin(dist)
if method == "knn":
    nn = np.argsort(dist)[:5]
    pred = np.bincount(db_labels[nn]).argmax()
    best = nn[db_labels[nn] == pred][0]

print("method:", method)
print("query class:", names[qc])
print("found class:", names[db_labels[best]])
print("distance:", dist[best])

plt.subplot(1, 2, 1)
plt.imshow(query, cmap="gray")
plt.title("query: " + names[qc])
plt.subplot(1, 2, 2)
plt.imshow(db_imgs[best], cmap="gray")
plt.title(f"{names[db_labels[best]]}, d={dist[best]:.4f}")
plt.show()
