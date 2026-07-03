import cv2
import numpy as np
import matplotlib.pyplot as plt

def make_test_grid(size=(400, 400)):
    grid = np.zeros((size[0], size[1], 3), dtype=np.uint8)

    # Draw green border
    cv2.rectangle(grid, (100, 100), (300, 300), (0, 255, 0), 2)

    # Grid lines
    for i in range(100, 350, 50):
        cv2.line(grid, (i, 100), (i, 300), (255, 255, 255), 1)
        cv2.line(grid, (100, i), (300, i), (255, 255, 255), 1)

    # Text marker
    cv2.putText(grid, 'F', (120, 180), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 0), 4)
    return grid

def get_2d_transforms(img):
    h, w, _ = img.shape
    out = {}

    # 1. Translation
    tx, ty = 40, -30
    m_trans = np.float32([[1, 0, tx],
                          [0, 1, ty]])
    out['1. Translation'] = cv2.warpAffine(img, m_trans, (w, h))

    # 2. Scaling
    sx, sy = 1.3, 0.7
    m_scale = np.float32([[sx, 0,  0],
                          [0,  sy, 0]])
    out['2. Scaling'] = cv2.warpAffine(img, m_scale, (w, h))

    # 3. Rotation (20 degrees)
    rad = np.radians(20)
    c, s = np.cos(rad), np.sin(rad)
    m_rot = np.float32([[c, -s, 0],
                        [s,  c, 0]])
    out['3. Rotation'] = cv2.warpAffine(img, m_rot, (w, h))

    # 4. Reflection X-axis
    m_ref_x = np.float32([[1,  0, 0],
                          [0, -1, h]])
    out['4. Reflection X'] = cv2.warpAffine(img, m_ref_x, (w, h))

    # 5. Reflection Y-axis
    m_ref_y = np.float32([[-1, 0, w],
                          [ 0, 1, 0]])
    out['5. Reflection Y'] = cv2.warpAffine(img, m_ref_y, (w, h))

    # 6. Reflection Origin
    m_ref_orig = np.float32([[-1,  0, w],
                             [ 0, -1, h]])
    out['6. Reflection Origin'] = cv2.warpAffine(img, m_ref_orig, (w, h))

    # 7. Reflection y = x
    m_ref_yx = np.float32([[0, 1, 0],
                           [1, 0, 0]])
    out['7. Reflection y=x'] = cv2.warpAffine(img, m_ref_yx, (w, h))

    # 8. X-Direction Shear
    shx = 0.25
    m_shear_x = np.float32([[1, shx, 0],
                            [0, 1,   0]])
    out['8. X-Shear'] = cv2.warpAffine(img, m_shear_x, (w, h))

    # 9. Y-Direction Shear
    shy = 0.20
    m_shear_y = np.float32([[1,   0, 0],
                            [shy, 1, 0]])
    out['9. Y-Shear'] = cv2.warpAffine(img, m_shear_y, (w, h))

    return out

if __name__ == '__main__':
    path = 'test_image.jpg'
    img_bgr = cv2.imread(path)

    if img_bgr is None:
        print(f"Image not found at {path}. Using generated F-grid.")
        img = make_test_grid()
    else:
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Run all transformations
    transforms = get_2d_transforms(img)

    titles = ['Original'] + list(transforms.keys())
    imgs = [img] + list(transforms.values())

    plt.figure(figsize=(15, 10))
    for i in range(len(imgs)):
        plt.subplot(2, 5, i + 1)
        plt.imshow(imgs[i])
        plt.title(titles[i], fontsize=10, fontweight='bold')
        plt.axis('on')
        plt.grid(True, linestyle=':', alpha=0.5)

    plt.suptitle("2D Affine and Geometric Transformations", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
