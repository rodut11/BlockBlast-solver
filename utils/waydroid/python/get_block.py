import ctypes
import numpy as np
import cv2
import sys

# ---------------- Load screencap ----------------
lib = ctypes.CDLL('../utils/waydroid/build/libwaydroid.so')
lib.get_screencap.argtypes = [ctypes.POINTER(ctypes.c_size_t)]
lib.get_screencap.restype = ctypes.POINTER(ctypes.c_ubyte)

size = ctypes.c_size_t()
buffer_ptr = lib.get_screencap(ctypes.byref(size))
img_data = np.ctypeslib.as_array(buffer_ptr, shape=(size.value,))
img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)

# ---------------- Specify block area ----------------
x1, y1, x2, y2 = 693, 711, 1183, 855
search_region = img[y1:y2, x1:x2]
gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# ---------------- Morphological closing + dilation ----------------
kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=1)
dilate_kernel = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)
cleaned = cv2.dilate(cleaned, dilate_kernel, iterations=2)

# ---------------- Remove small noise blobs ----------------
def remove_small_components(mask, min_area=40):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for i in range(1, num_labels):
        _, _, _, _, area = stats[i]
        if area >= min_area:
            cleaned[labels == i] = 255
    return cleaned

cleaned = remove_small_components(cleaned, min_area=40)

# ---------------- Load single-cell template ----------------
cell_template = cv2.imread("/home/rodut11/Coding/C/Block-blast-solver/assets/sample.png", cv2.IMREAD_GRAYSCALE)
_, cell_template = cv2.threshold(cell_template, 127, 255, cv2.THRESH_BINARY)
cell_h, cell_w = cell_template.shape

# ---------------- Connected components ----------------
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
    h_crop = n_rows * cell_h
    w_crop = n_cols * cell_w
    cropped = cropped[:h_crop, :w_crop]

    grid = np.zeros((n_rows, n_cols), dtype=np.uint8)
    for r in range(n_rows):
        for c in range(n_cols):
            y1, y2 = r * cell_h, (r+1) * cell_h
            x1, x2 = c * cell_w, (c+1) * cell_w
            cell = cropped[y1:y2, x1:x2]
            filled_ratio = np.sum(cell > 0) / (cell.size + 1e-6)
            if filled_ratio > fill_ratio_thresh:
                grid[r, c] = 1

    return grid, []

# ---------------- Detect and output first 3 block grids ----------------
ROW_SEPARATOR = b"\xFE"
BLOCK_SEPARATOR = b"\xFF"
HEADER_MARKER  = b"\xFD"  # marker to signify end of header

merged_boxes = get_block_boxes(cleaned, min_area=64)[:3]

# --- Write header bytes: rows/cols of first 3 blocks ---
header_bytes = []
grids = []

for block_img in [cleaned[y:y+h, x:x+w] for (x,y,w,h) in merged_boxes]:
    grid, _ = detect_cells_grid(block_img, cell_w, cell_h)
    rows, cols = grid.shape
    header_bytes.extend([rows, cols])
    grids.append(grid)

# Write first 6 bytes (rows/cols for 3 blocks) + marker
sys.stdout.buffer.write(bytes(header_bytes) + HEADER_MARKER)

# --- Write actual blocks with row & block separators ---
for grid in grids:
    rows, cols = grid.shape
    for r in range(rows):
        sys.stdout.buffer.write(grid[r, :].tobytes())
        sys.stdout.buffer.write(ROW_SEPARATOR)
    sys.stdout.buffer.write(BLOCK_SEPARATOR)

