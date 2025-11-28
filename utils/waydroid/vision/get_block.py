import ctypes
import numpy as np
import cv2
import sys
import os

# --- Script folder + paths ---
script_dir = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(script_dir, '..', 'build', 'get_screencap.so')
project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
sample_path = os.path.join(project_root, 'assets', 'sample.png')
# --- Load screencap ---
lib = ctypes.CDLL(lib_path)
lib.get_screencap.argtypes = [ctypes.POINTER(ctypes.c_size_t)]
lib.get_screencap.restype = ctypes.POINTER(ctypes.c_ubyte)

size = ctypes.c_size_t()
buffer_ptr = lib.get_screencap(ctypes.byref(size))
img_data = np.ctypeslib.as_array(buffer_ptr, shape=(size.value,))
img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)

# --- Block region ---
x1, y1, x2, y2 = 693, 711, 1183, 855
search_region = img[y1:y2, x1:x2]
gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# --- Morphological cleaning ---
kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=1)
dilate_kernel = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)
cleaned = cv2.dilate(cleaned, dilate_kernel, iterations=2)

# --- Remove small blobs ---
def remove_small_components(mask, min_area=40):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, 4] >= min_area:
            out[labels == i] = 255
    return out

cleaned = remove_small_components(cleaned, min_area=40)

# --- Load single-cell template ---
cell_template = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)
if cell_template is None:
    raise FileNotFoundError(f"Cannot find {sample_path}")
_, cell_template = cv2.threshold(cell_template, 127, 255, cv2.THRESH_BINARY)
cell_h, cell_w = cell_template.shape

# --- Connected components ---
def get_block_boxes(mask, min_area=64):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area >= min_area:
            boxes.append((x, y, w, h))
    return boxes

def detect_cells_grid(block_img, cell_w, cell_h, fill_ratio_thresh=0.55):
    ys, xs = np.where(block_img > 0)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros((1,1), dtype=np.uint8), []

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    cropped = block_img[y_min:y_max+1, x_min:x_max+1]

    h, w = cropped.shape
    n_rows = max(1, h // cell_h)
    n_cols = max(1, w // cell_w)
    cropped = cropped[:n_rows*cell_h, :n_cols*cell_w]

    grid = np.zeros((n_rows, n_cols), dtype=np.uint8)
    for r in range(n_rows):
        for c in range(n_cols):
            cell = cropped[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
            res = cv2.matchTemplate(cell, cell_template, cv2.TM_CCOEFF_NORMED)
            if res.max() > threshold:
                grid[r, c] = 1


    return grid, []

# --- Output ---
ROW_SEPARATOR = b"\xFE"
BLOCK_SEPARATOR = b"\xFF"
HEADER_MARKER  = b"\xFD"

merged_boxes = sorted(get_block_boxes(cleaned, min_area=64), key=lambda b: b[0])[:3]

header_bytes = []
grids = []

for block_img in [cleaned[y:y+h, x:x+w] for (x,y,w,h) in merged_boxes]:
    grid, _ = detect_cells_grid(block_img, cell_w, cell_h)
    rows, cols = grid.shape
    header_bytes.extend([rows, cols])
    grids.append(grid)

sys.stdout.buffer.write(bytes(header_bytes) + HEADER_MARKER)

for grid in grids:
    for r in range(grid.shape[0]):
        sys.stdout.buffer.write(grid[r, :].tobytes())
        sys.stdout.buffer.write(ROW_SEPARATOR)
    sys.stdout.buffer.write(BLOCK_SEPARATOR)