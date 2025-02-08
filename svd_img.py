import numpy as np
import matplotlib.pyplot as plt
# Скачивание картинки.
from PIL import Image, ImageOps
import requests

# Ссылка на картинку.
url = "https://code.s3.yandex.net/Math/images/carpet.jpeg"

# Скачивание и сохранение картинки в переменную.
im = ImageOps.grayscale(Image.open(requests.get(url, stream=True).raw))
# Превращение картинки в NumPy-массив.
carpet = np.array(im).astype(float)


def show_image(img):
	plt.figure(figsize=(12, 8))
	plt.imshow(img, cmap='gray')
	plt.xticks([])
	plt.yticks([])

show_image(carpet)

# Compress leaving 10 components and compute MSE
