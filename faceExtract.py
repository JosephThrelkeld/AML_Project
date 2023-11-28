from PIL import Image
import numpy as np

im = Image.open("att_faces\s1\\1.pgm")
im = im.convert("L")

data = np.asarray(im)\

print(data)