import ctypes
import numpy as np
import cv2

# load shared library
lib = ctypes.CDLL('utils/waydroid/build/get_screencap.so')  # replace with your shared library path

# tell python the function signature
lib.get_screencap.argtypes = [ctypes.POINTER(ctypes.c_size_t)]
lib.get_screencap.restype = ctypes.POINTER(ctypes.c_ubyte)

# prepare a size_t variable to receive the image size
size = ctypes.c_size_t()

# call the function
buffer_ptr = lib.get_screencap(ctypes.byref(size))

# convert to a numpy array
img_data = np.ctypeslib.as_array(buffer_ptr, shape=(size.value,))

# decode PNG in memory using OpenCV
img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)

# Save raw image to a file
with open("screencap.png", "wb") as f:
    f.write(img_data.tobytes())

# show the image
cv2.namedWindow('Screen', cv2.WINDOW_NORMAL)  # makes the window resizable

cv2.imshow('Screen', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


#x1, y1 = 693, 711  # top-left corner
# x2, y2 = 1183, 855  # bottom-right corner