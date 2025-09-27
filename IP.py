# Radiomics-style figure from the second image, with explanation in text below.
# The figure shows:
# (a) Original image with selected ROI (optimized rectangle) overlay
# (b) Zoomed ROI with a discrete gray-level grid (labels 1..Ng)
# (c) GLCM matrix heatmap for 0° (horizontal, distance=1)
# (d) GLRLM matrix heatmap for horizontal runs
# (e) Histogram of ROI intensities
#
# We recover the rectangle from the green border in the provided "optimized rectangle" image.

import numpy as np, cv2, matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib import gridspec

# ---------- helpers ----------
def find_rectangle_from_green_border(rgb):
    """Detect the bright green rectangle and return polygon points (4x2 float32)."""
    b,g,r = cv2.split(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))  # OpenCV BGR order
    # Threshold for green stroke (bright G, relatively low R/B)
    mask = (g > 160) & (r < 140) & (b < 140)
    mask = mask.astype(np.uint8) * 255
    # Clean and thicken a bit to stabilize
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    # Find contours
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, mask
    cnt = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)  # (center,(w,h), angle)
    box = cv2.boxPoints(rect)    # 4x2 float
    return box.astype(np.float32), mask

def polygon_mask(shape, poly):
    H, W = shape
    m = np.zeros((H, W), dtype=np.uint8)
    cv2.fillConvexPoly(m, poly.astype(np.int32), 1)
    return m

def crop_to_poly_bbox(img, poly, pad=4):
    x,y,w,h = cv2.boundingRect(poly.astype(np.int32))
    x = max(0, x-pad); y = max(0, y-pad)
    xe = min(img.shape[1], x+w+2*pad); ye = min(img.shape[0], y+h+2*pad)
    return img[y:ye, x:xe], (x,y,xe,ye)

def quantize_levels(img, mask, Ng=4):
    vals = img[mask>0]
    if vals.size == 0:
        bins = np.linspace(0, 1, Ng+1)
    else:
        vmin, vmax = float(np.min(vals)), float(np.max(vals))
        if vmax <= vmin:
            bins = np.linspace(vmin, vmin+1e-6, Ng+1)
        else:
            bins = np.linspace(vmin, vmax, Ng+1)
    q = np.digitize(img, bins[1:-1]) + 1  # 1..Ng
    q[mask==0] = 0
    return q, bins

def compute_glcm(img_q, mask, Ng=4, dx=1, dy=0):
    """GLCM for a single offset. img_q values should be 1..Ng, 0 for background (ignored)."""
    H, W = img_q.shape
    glcm = np.zeros((Ng, Ng), dtype=np.int64)
    # shifted coordinates
    x0 = max(0, +dx); x1 = W + min(0, +dx)
    y0 = max(0, +dy); y1 = H + min(0, +dy)
    A = img_q[y0:y1, x0:x1]
    B = img_q[y0-dy:y1-dy, x0-dx:x1-dx]
    M = mask[y0:y1, x0:x1] & mask[y0-dy:y1-dy, x0-dx:x1-dx]
    # valid where both inside mask and non-zero labels
    valid = (A>0) & (B>0) & (M>0)
    a = A[valid]-1; b = B[valid]-1  # 0..Ng-1
    # accumulate
    for i in range(a.size):
        glcm[a[i], b[i]] += 1
    return glcm

def compute_glrlm_horizontal(img_q, mask, Ng=4):
    """GLRLM for horizontal direction. Columns = run-length (1..Lmax)."""
    H, W = img_q.shape
    Lmax = W
    rl = np.zeros((Ng, Lmax), dtype=np.int64)
    for y in range(H):
        row = img_q[y]
        row_m = mask[y]>0
        run_val, run_len = 0, 0
        for x in range(W+1):  # sentinel at W
            v = row[x] if x<W else -1
            m = row_m[x] if x<W else False
            if m and v>0 and (run_len==0 or v==run_val):
                # continue run
                run_val = v; run_len += 1
            else:
                # end run if existed
                if run_len>0 and 1 <= run_val <= Ng:
                    rl[run_val-1, run_len-1] += 1
                # start new if valid
                if x<W and m and v>0:
                    run_val, run_len = v, 1
                else:
                    run_val, run_len = 0, 0
    # trim zero columns to max non-zero
    if rl.sum()>0:
        nonzero_cols = np.where(rl.sum(0)>0)[0]
        if nonzero_cols.size>0:
            rl = rl[:, :nonzero_cols[-1]+1]
    return rl

# ---------- load image ----------
path = "/mnt/data/4eb5454a-3f6f-4881-bf3f-56f4c290f3d4.png"  # "Optimized rectangle"
img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
if img_bgr is None:
    raise FileNotFoundError(path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
img_gray = (img_gray - img_gray.min()) / (img_gray.ptp() + 1e-6)  # 0..1

# detect polygon
poly, green_mask = find_rectangle_from_green_border(img_rgb)
if poly is None:
    # fallback: center box
    H, W = img_gray.shape
    w,h = int(0.5*W), int(0.4*H)
    cx, cy = W//2, H//2
    poly = np.array([[cx-w//2, cy-h//2],[cx+w//2, cy-h//2],[cx+w//2, cy+h//2],[cx-w//2, cy+h//2]], dtype=np.float32)

# interior mask (eroded to avoid border contamination)
mask_poly = polygon_mask(img_gray.shape, poly)
mask_interior = cv2.erode(mask_poly, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))

# cropped ROI
roi_gray, (x0,y0,xe,ye) = crop_to_poly_bbox(img_gray, poly, pad=4)
roi_mask = mask_interior[y0:ye, x0:xe]

# quantize to Ng levels
Ng = 4
roi_q, bins = quantize_levels(roi_gray, roi_mask, Ng=Ng)

# grid overlay data for (b): compute a coarse grid of labels
cell = max(8, int(min(roi_gray.shape)/10))  # choose a reasonable cell size
Hc, Wc = roi_gray.shape
grid_labels = np.zeros_like(roi_q)
for y in range(0, Hc, cell):
    for x in range(0, Wc, cell):
        ys, ye2 = y, min(Hc, y+cell)
        xs, xe2 = x, min(Wc, x+cell)
        block = roi_q[ys:ye2, xs:xe2]
        block_m = roi_mask[ys:ye2, xs:xe2]>0
        if block_m.any():
            vals = block[block_m]
            # choose the most frequent label
            lbl = np.bincount(vals, minlength=Ng+1).argmax()
            grid_labels[ys:ye2, xs:xe2] = lbl

# compute GLCM (0°) and GLRLM (horizontal)
glcm = compute_glcm(roi_q, roi_mask, Ng=Ng, dx=1, dy=0)
glrlm = compute_glrlm_horizontal(roi_q, roi_mask, Ng=Ng)

# histogram of ROI intensities
hist_vals = roi_gray[roi_mask>0]

# ---------- figure layout ----------
fig = plt.figure(figsize=(12,7))
gs = gridspec.GridSpec(2, 3, width_ratios=[1.2, 1.0, 1.0], height_ratios=[1.0, 1.2])
gs.update(wspace=0.45, hspace=0.4)

# (a) full image + ROI overlay
ax_a = fig.add_subplot(gs[0,0])
ax_a.imshow(img_gray, cmap='gray')
ax_a.fill(poly[:,0], poly[:,1], edgecolor='lime', fill=False, linewidth=4)
ax_a.set_title("a) Original with ROI")
ax_a.axis('off')
# draw dashed zoom box
ax_a.add_patch(Rectangle((x0,y0), xe-x0, ye-y0, fill=False, linestyle='--', linewidth=2))

# (b) zoomed ROI + grid labels
ax_b = fig.add_subplot(gs[0,1])
ax_b.imshow(roi_gray, cmap='gray')
# overlay grid
for x in range(0, Wc, cell):
    ax_b.axvline(x-0.5, color='white', linestyle=':', linewidth=0.8, alpha=0.8)
for y in range(0, Hc, cell):
    ax_b.axhline(y-0.5, color='white', linestyle=':', linewidth=0.8, alpha=0.8)
# numbers
for y in range(0, Hc, cell):
    for x in range(0, Wc, cell):
        ys, ye2 = y, min(Hc, y+cell)
        xs, xe2 = x, min(Wc, x+cell)
        sub = grid_labels[ys:ye2, xs:xe2]
        lbl = int(np.round(np.mean(sub[sub>0]))) if (sub>0).any() else 0
        if lbl>0:
            ax_b.text(x+cell/2, y+cell/2, f"{lbl}", ha='center', va='center', fontsize=10,
                      color='white', weight='bold')
ax_b.set_title("b) Zoomed ROI with quantized grid (1..4)")
ax_b.axis('off')

# (c) GLCM heatmap
ax_c = fig.add_subplot(gs[0,2])
im_c = ax_c.imshow(glcm, interpolation='nearest')
ax_c.set_title("c) GLCM (0°, d=1)")
ax_c.set_xlabel("j")
ax_c.set_ylabel("i")
ax_c.set_xticks(range(Ng)); ax_c.set_yticks(range(Ng))
ax_c.set_xticklabels(range(1,Ng+1)); ax_c.set_yticklabels(range(1,Ng+1))
fig.colorbar(im_c, ax=ax_c, fraction=0.046, pad=0.04)

# (d) GLRLM heatmap
ax_d = fig.add_subplot(gs[1,1])
im_d = ax_d.imshow(glrlm, interpolation='nearest', aspect='auto')
ax_d.set_title("d) GLRLM (horizontal)")
ax_d.set_xlabel("Run length")
ax_d.set_ylabel("Gray level")
ax_d.set_yticks(range(Ng)); ax_d.set_yticklabels(range(1,Ng+1))
fig.colorbar(im_d, ax=ax_d, fraction=0.046, pad=0.04)

# (e) Histogram
ax_e = fig.add_subplot(gs[1,2])
ax_e.hist(hist_vals.ravel(), bins=32)
ax_e.set_title("e) ROI Intensity Histogram")
ax_e.set_xlabel("Intensity")
ax_e.set_ylabel("Count")

# arrows from (a) to (b), (b) to (c,d,e)
# compute axes centers
def axes_center(ax):
    x0,x1 = ax.get_position().x0, ax.get_position().x1
    y0,y1 = ax.get_position().y0, ax.get_position().y1
    return (0.5*(x0+x1), 0.5*(y0+y1))

c_a = axes_center(ax_a); c_b = axes_center(ax_b)
c_c = axes_center(ax_c); c_d = axes_center(ax_d); c_e = axes_center(ax_e)

for (p,q) in [(c_a,c_b),(c_b,c_c),(c_b,c_d),(c_b,c_e)]:
    fig.add_artist(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=10, lw=1.0, linestyle='--'))

fig.suptitle("Radiomics-style illustration from optimized rectangle", y=0.98, fontsize=14)
out_path = "/mnt/data/radiomics_diagram_from_rectangle.png"
plt.savefig(out_path, dpi=200, bbox_inches='tight')
out_path
