import ctypes
import numpy as np
import cv2
import json

# ---------------- Load screencap ----------------
lib = ctypes.CDLL('utils/waydroid/build/libwaydroid.so')
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

# ---------------- Morphological closing + aggressive dilation ----------------
kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=1)

dilate_kernel = np.array([[0,1,0],
                          [1,1,1],
                          [0,1,0]], dtype=np.uint8)
cleaned = cv2.dilate(cleaned, dilate_kernel, iterations=2)

# ---- REMOVE SMALL NOISE BLOBS ----
def remove_small_components(mask, min_area=40):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for i in range(1, num_labels):
        _, _, _, _, area = stats[i]
        if area >= min_area:
            cleaned[labels == i] = 255
    return cleaned

cleaned = remove_small_components(cleaned, min_area=40)

# ---------------- Load block definitions ----------------
with open("blocks.json", "r") as f:
    blocks = json.load(f)

# ---------------- Load single-cell template ----------------
cell_template = cv2.imread("assets/sample.png", cv2.IMREAD_GRAYSCALE)
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
    """Divide block into grid cells and detect filled cells."""
    ys, xs = np.where(block_img > 0)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros((1,1), dtype=int), []

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    cropped = block_img[y_min:y_max+1, x_min:x_max+1]

    h, w = cropped.shape
    n_rows = max(1, h // cell_h)
    n_cols = max(1, w // cell_w)
    h_crop = n_rows * cell_h
    w_crop = n_cols * cell_w
    cropped = cropped[:h_crop, :w_crop]

    grid = np.zeros((n_rows, n_cols), dtype=int)
    centers = []

    for r in range(n_rows):
        for c in range(n_cols):
            y1 = r * cell_h
            y2 = y1 + cell_h
            x1 = c * cell_w
            x2 = x1 + cell_w
            cell = cropped[y1:y2, x1:x2]
            filled_ratio = np.sum(cell > 0) / (cell.size + 1e-6)

            if filled_ratio > fill_ratio_thresh:
                grid[r, c] = 1
                cx = x_min + x1 + cell_w // 2
                cy = y_min + y1 + cell_h // 2
                centers.append((cx, cy))

    # ---- REMOVE ISOLATED SINGLE PIXELS (false dots) ----
    kernel = np.ones((3, 3), np.uint8)
    neighbor_count = cv2.filter2D(grid.astype(np.uint8), -1, kernel)
    grid[(grid == 1) & (neighbor_count <= 2)] = 0

    # filter centers accordingly
    centers = [(x_min + c * cell_w + cell_w // 2, y_min + r * cell_h + cell_h // 2)
               for r in range(grid.shape[0]) for c in range(grid.shape[1]) if grid[r, c] == 1]

    return grid, centers


def match_grid(grid, blocks):
    for blk in blocks:
        blk_pat = np.array(blk["pattern"])
        if grid.shape == blk_pat.shape and np.array_equal(grid, blk_pat):
            return blk["name"]
    return "unknown"


def print_grid(grid):
    for row in grid:
        print(" ".join(str(c) for c in row))

# ---------------- Detect and classify blocks ----------------
merged_boxes = get_block_boxes(cleaned, min_area=64)
vis = search_region.copy()

for i, (x, y, w, h) in enumerate(merged_boxes):
    block_img = cleaned[y:y+h, x:x+w]
    grid, centers = detect_cells_grid(block_img, cell_w, cell_h)
    name = match_grid(grid, blocks)

    cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(vis, name, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    print(f"Block {i+1} at ({x},{y}) size ({w}x{h}) classified as {name}")
    print_grid(grid)
    print()

    # centers already include x_min/y_min, so just offset by the block’s position
    for cx, cy in centers:
        cv2.circle(vis, (x + cx, y + cy), 3, (0, 0, 255), -1)

cv2.imshow("Detected and Classified Blocks", vis)
cv2.waitKey(0)
cv2.destroyAllWindows()

