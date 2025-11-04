import numpy as np
import cv2
from tkinter import Tk, filedialog, messagebox
import matplotlib.pyplot as plt
import os


RESIZE = (64, 64)
BLOCKS = (16, 16)
BIN_THRESHOLD = 0.5
MAX_ITER = 50
V = 0.05  


def choose_files(multiple=True, title="Виберіть файл(и)"):
    root = Tk()
    root.withdraw()
    files = filedialog.askopenfilenames(title=title) if multiple else (filedialog.askopenfilename(title=title),)
    root.destroy()
    return list(files)


def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: raise FileNotFoundError(path)
    return img.astype(np.float32)


def compute_block_means(img):
    img_small = cv2.resize(img, RESIZE, interpolation=cv2.INTER_AREA)
    h, w = RESIZE
    bh, bw = h // BLOCKS[0], w // BLOCKS[1]
    feats = []
    for i in range(BLOCKS[0]):
        for j in range(BLOCKS[1]):
            feats.append(img_small[i*bh:(i+1)*bh, j*bw:(j+1)*bw].mean())
    return np.array(feats, dtype=np.float32)

#  Мінмакс нормалізація 
def normalize_minmax(vec):
    vmin, vmax = vec.min(), vec.max()
    return np.zeros_like(vec) if np.isclose(vmin, vmax) else (vec - vmin) / (vmax - vmin)

# Перетворення у біполярний вектор 
def to_bipolar(vec, threshold=BIN_THRESHOLD):
    return np.where(vec >= threshold, 1, -1).astype(np.int8)


def first_layer_weights(patterns, all_files):
    """
    W_ik = mean(x_ik)/2
    B_k = n/2
    """
    num_classes = len(all_files)
    n_features = patterns.shape[1]
    W1 = np.zeros((n_features, num_classes), dtype=np.float32)
    B = np.zeros(num_classes, dtype=np.float32)
    
    start = 0
    for k, group in enumerate(all_files):
        count = len(group)
        class_patterns = patterns[start:start+count]
        W1[:, k] = class_patterns.mean(axis=0) / 2
        B[k] = n_features / 2  # поріг B_k
        start += count
    return W1, B

def first_layer_output(W1, B, x):
    y = W1.T.dot(x)
    # застосування порога B_k
    return np.where(y >= B, 1, -1).astype(np.int8)


def init_second_layer(num_classes):
    W2 = np.full((num_classes, num_classes), -V)
    np.fill_diagonal(W2, 1.0)
    return W2

def second_layer_recall(W2, y_init, max_iter=MAX_ITER):
    y = y_init.copy()
    for _ in range(max_iter):
        y_new = np.sign(W2.dot(y))
        if np.array_equal(y, y_new):
            break
        y = y_new
    return y

# --- Генератор тестових зображень ---
def generate_test_images(num_classes=3, num_per_class=3, save_dir="test_images"):
    os.makedirs(save_dir, exist_ok=True)
    for cls in range(num_classes):
        for idx in range(num_per_class):
            img = np.random.randint(0, 256, RESIZE, dtype=np.uint8)
            filename = os.path.join(save_dir, f"class{cls+1}_{idx+1}.png")
            cv2.imwrite(filename, img)
    print(f"Згенеровано {num_classes*num_per_class} тестових зображень у {save_dir}")


def main():
    
    all_files = []
    for cls in ['A','B','C']:
        files = choose_files(title=f"Файли класу {cls}")
        if not files: return
        all_files.append(files)

    patterns = []
    for group in all_files:
        for f in group:
            img = load_image(f)
            vec = normalize_minmax(compute_block_means(img))
            patterns.append(to_bipolar(vec))
    patterns = np.array(patterns)

    #  Перший прошарок 
    W1, B = first_layer_weights(patterns, all_files)
    first_outputs = np.array([first_layer_output(W1, B, p) for p in patterns])

    #  Другий прошарок 
    num_classes = len(all_files)
    W2 = init_second_layer(num_classes)
    class_outputs = np.array([second_layer_recall(W2, first_outputs[i]) for i in range(len(patterns))])

    
    unk_file = choose_files(multiple=False, title="Невідомий образ")
    if not unk_file: return
    uimg = load_image(unk_file[0])
    uvec = to_bipolar(normalize_minmax(compute_block_means(uimg)))
    y_init = first_layer_output(W1, B, uvec)
    y_class = second_layer_recall(W2, y_init)
    
    predicted_class = np.argmax(y_class) + 1
    messagebox.showinfo("Результат", f"Найближчий клас: {predicted_class}")

if __name__ == "__main__":
    main()
