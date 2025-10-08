import ctypes
import numpy as np
import cv2
import sys

# --- Load screencap ---
lib = ctypes.CDLL('../utils/waydroid/build/libwaydroid.so')
lib.get_screencap.argtypes = [ctypes.POINTER(ctypes.c_size_t)]
lib.get_screencap.restype = ctypes.POINTER(ctypes.c_ubyte)

size = ctypes.c_size_t()
buffer_ptr = lib.get_screencap(ctypes.byref(size))
img_data = np.ctypeslib.as_array(buffer_ptr, shape=(size.value,))
img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)

# --- Define grid area ---
x1, y1, x2, y2 = 716, 206, 1163, 654
region = img[y1:y2, x1:x2]
gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# --- Remove small junk ---
def remove_small_components(mask, min_area=40):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for i in range(1, num_labels):
        _, _, _, _, area = stats[i]
        if area >= min_area:
            cleaned[labels == i] = 255
    return cleaned

cleaned = remove_small_components(binary, min_area=40)

# --- Divide region into 8x8 grid and detect filled cells ---
h, w = cleaned.shape
cell_h, cell_w = h // 8, w // 8

grid = np.zeros((8, 8), dtype=np.uint8)
for r in range(8):
    for c in range(8):
        y1, y2 = r * cell_h, (r + 1) * cell_h
        x1, x2 = c * cell_w, (c + 1) * cell_w
        cell = cleaned[y1:y2, x1:x2]
        filled_ratio = np.sum(cell > 0) / (cell.size + 1e-6)
        if filled_ratio > 0.5:
            grid[r, c] = 1

# --- Build binary stream with end-of-row markers (0xFF) ---
output_bytes = bytearray()
for r in range(8):
    output_bytes.extend(grid[r].tobytes())
    output_bytes.append(0xFE)  # End of row marker

# --- Output raw binary ---
sys.stdout.buffer.write(output_bytes)
