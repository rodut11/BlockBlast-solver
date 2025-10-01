import ctypes
import numpy as np
import cv2

# load shared library
lib = ctypes.CDLL('utils/waydroid/build/libwaydroid.so')

lib.get_screencap.argtypes = [ctypes.POINTER(ctypes.c_size_t)]
lib.get_screencap.restype = ctypes.POINTER(ctypes.c_ubyte)

size = ctypes.c_size_t()
buffer_ptr = lib.get_screencap(ctypes.byref(size))
img_data = np.ctypeslib.as_array(buffer_ptr, shape=(size.value,))
img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)

# -------- define search region --------
x1, y1, x2, y2 = 693, 711, 1183, 855
search_region = img[y1:y2, x1:x2]

# convert region to grayscale
gray_region = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)

# -------- load + preprocess sample --------
sample = cv2.imread("assets/sample.png", cv2.IMREAD_GRAYSCALE)
if sample is None:
    raise FileNotFoundError("sample.png not found!")

# -------- template matching --------
res = cv2.matchTemplate(gray_region, sample, cv2.TM_CCOEFF_NORMED)
`
# similarity threshold
threshold = 0.78
loc = np.where(res >= threshold)

h, w = sample.shape[::-1]
# make 3-channel grayscale so we can draw colored rectangles
vis = cv2.cvtColor(gray_region, cv2.COLOR_GRAY2BGR)

for pt in zip(*loc[::-1]):  # (x,y)
    cv2.rectangle(vis, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)

# show results
cv2.imshow("Detected Blocks", vis)
cv2.waitKey(0)
cv2.destroyAllWindows()
