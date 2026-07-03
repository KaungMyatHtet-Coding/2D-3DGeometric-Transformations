import cv2
import numpy as np
import matplotlib.pyplot as plt

def make_grid(size=(400, 400)):
    # Black background
    grid = np.zeros((size[0], size[1], 3), dtype=np.uint8)

    # Draw green border
    cv2.rectangle(grid, (50, 50), (350, 350), (0, 255, 0), 2)

    # Inner grid lines
    for i in range(50, 350, 50):
        cv2.line(grid, (i, 50), (i, 350), (255, 255, 255), 1)
        cv2.line(grid, (50, i), (350, i), (255, 255, 255), 1)

    # Text label
    cv2.putText(grid, 'UIT CV', (70, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
    return grid

def do_2d_transforms(img):
    h, w, _ = img.shape
    out = {}

    # 1. Translation (2 DoF)
    tx, ty = 40, 30
    mat_t = np.float32([[1, 0, tx],
                         [0, 1, ty]])
    out['Translation'] = cv2.warpAffine(img, mat_t, (w, h))

    # 2. Rigid / Euclidean (3 DoF)
    deg = np.radians(15)
    c, s = np.cos(deg), np.sin(deg)
    mat_r = np.float32([[c, -s, 20],
                         [s,  c, 10]])
    out['Rigid'] = cv2.warpAffine(img, mat_r, (w, h))

    # 3. Similarity (4 DoF)
    scale = 0.75
    mat_s = np.float32([[scale * c, -scale * s, 60],
                         [scale * s,  scale * c, 40]])
    out['Similarity'] = cv2.warpAffine(img, mat_s, (w, h))

    # 4. Affine Transformation (6 DoF)
    p_src = np.float32([[50, 50], [200, 50], [50, 200]])
    p_dst = np.float32([[70, 60], [180, 80], [60, 230]])
    mat_a = cv2.getAffineTransform(p_src, p_dst)
    out['Affine'] = cv2.warpAffine(img, mat_a, (w, h))

    # 5. Projective / Homography (8 DoF)
    p_src4 = np.float32([[50, 50], [350, 50], [50, 350], [350, 350]])
    p_dst4 = np.float32([[80, 70], [320, 100], [40, 320], [380, 280]])
    mat_p = cv2.getPerspectiveTransform(p_src4, p_dst4)
    out['Projective'] = cv2.warpPerspective(img, mat_p, (w, h))

    return out

def check_3d_transforms():
    print("\n--- 3D Transformations Check ---")
    # Target 3D point in homogeneous coords
    P = np.array([10.0, 20.0, 30.0, 1.0])
    print("Original P:", P[:3])

    # 1. 3D Translation
    tx, ty, tz = 5, -10, 15
    T = np.array([[1, 0, 0, tx],
                  [0, 1, 0, ty],
                  [0, 0, 1, tz],
                  [0, 0, 0, 1]])
    print("3D Translated:", (T @ P)[:3])

    # 2. 3D Scaling
    sx, sy, sz = 1.5, 2.0, 0.5
    S = np.array([[sx,  0,  0, 0],
                  [ 0, sy,  0, 0],
                  [ 0,  0, sz, 0],
                  [ 0,  0,  0, 1]])
    print("3D Scaled:", (S @ P)[:3])

    # 3. 3D Rotations (45 degrees)
    rad = np.radians(45)
    ca, sa = np.cos(rad), np.sin(rad)

    # Z-axis
    R_z = np.array([[ca, -sa,  0,  0],
                    [sa,  ca,  0,  0],
                    [ 0,   0,  1,  0],
                    [ 0,   0,  0,  1]])

    # Y-axis
    R_y = np.array([[ ca,  0, sa,  0],
                    [  0,  1,  0,  0],
                    [-sa,  0, ca,  0],
                    [  0,  0,  0,  1]])

    # X-axis
    R_x = np.array([[1,   0,   0,  0],
                    [0,  ca, -sa,  0],
                    [0,  sa,  ca,  0],
                    [0,   0,   0,  1]])

    print("Rotate X:", (R_x @ P)[:3])
    print("Rotate Y:", (R_y @ P)[:3])
    print("Rotate Z:", (R_z @ P)[:3])

if __name__ == '__main__':
    path = 'test_image.jpg'
    img_bgr = cv2.imread(path)

    if img_bgr is None:
        print(f"Image not found at {path}. Using generated grid template.")
        img = make_grid()
    else:
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Run transforms
    results = do_2d_transforms(img)

    titles = ['Original'] + list(results.keys())
    imgs = [img] + list(results.values())

    plt.figure(figsize=(14, 9))
    for i in range(len(imgs)):
        plt.subplot(2, 3, i + 1)
        plt.imshow(imgs[i])
        plt.title(titles[i], fontsize=11, fontweight='bold')
        plt.axis('on')
        plt.grid(True, linestyle=':', alpha=0.6)

    plt.suptitle("2D / 3D Geometric Transformations", fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Print 3D numerical output
    check_3d_transforms()

    plt.show()
