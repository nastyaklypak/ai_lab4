import numpy as np
import cv2
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog, messagebox

RESIZE = (64, 64)
BLOCKS = (8, 8)
LEARNING_RATE = 0.1
EPOCHS = 100


def load_image(path, gray=True):
    flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
    img = cv2.imread(path, flag)
    if img is None:
        raise FileNotFoundError(f"Не можу відкрити {path}")
    return img.astype(np.float32)


def compute_block_means(img, resize=RESIZE, blocks=BLOCKS):
    img_r = cv2.resize(img, resize, interpolation=cv2.INTER_AREA)
    h, w = resize
    bh = h // blocks[0]
    bw = w // blocks[1]
    
    feats = []
    for i in range(blocks[0]):
        for j in range(blocks[1]):
            y0, y1 = i*bh, (i+1)*bh
            x0, x1 = j*bw, (j+1)*bw
            block = img_r[y0:y1, x0:x1]
            feats.append(block.mean())
    return np.array(feats, dtype=np.float32)


def normalize_minmax(vec):
    vmin = vec.min()
    vmax = vec.max()
    if np.isclose(vmax, vmin):
        return np.zeros_like(vec)
    return (vec - vmin) / (vmax - vmin)


def choose_files(multiple=True, title="Виберіть файл(и)"):
    root = Tk()
    root.withdraw()
    if multiple:
        files = filedialog.askopenfilenames(title=title)
    else:
        f = filedialog.askopenfilename(title=title)
        files = (f,) if f else ()
    root.destroy()
    return list(files)


def display_images_with_titles(img_list, titles, cmap='gray'):
    n = len(img_list)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(4*cols, 3*rows))
    for i, img in enumerate(img_list):
        plt.subplot(rows, cols, i+1)
        plt.imshow(img, cmap=cmap)
        plt.title(titles[i], fontsize=10)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


class Perceptron:
    def __init__(self, input_size, lr=LEARNING_RATE):
        self.w = np.random.rand(input_size)  # ваги
        self.b = 0.0  # зсув
        self.lr = lr

    def activation(self, x):
        return 1 if x >= 0 else -1

    def predict(self, x):
        return self.activation(np.dot(self.w, x) + self.b)

    def train(self, X, y, epochs=EPOCHS):
        for _ in range(epochs):
            for xi, yi in zip(X, y):
                y_pred = self.predict(xi)
                if y_pred != yi:
                    self.w += self.lr * (yi - y_pred) * xi
                    self.b += self.lr * (yi - y_pred)


def main():
    # Вибір файлів для класу A (+1)
    print("Виберіть файли класу A (+1)")
    class_a_files = choose_files(multiple=True, title="Виберіть файли класу A (+1)")
    if len(class_a_files) < 1:
        print("Файли класу A не вибрано. Вихід.")
        return

    # Вибір файлів для класу B (-1)
    print("Виберіть файли класу B (-1)")
    class_b_files = choose_files(multiple=True, title="Виберіть файли класу B (-1)")
    if len(class_b_files) < 1:
        print("Файли класу B не вибрано. Вихід.")
        return

    all_files = class_a_files + class_b_files
    images, abs_vecs, norm_vecs, labels = [], [], [], []

   
    for f in class_a_files:
        img = load_image(f)
        abs_vec = compute_block_means(img)
        norm_vec = normalize_minmax(abs_vec)
        images.append(img)
        abs_vecs.append(abs_vec)
        norm_vecs.append(norm_vec)
        labels.append(1)

    for f in class_b_files:
        img = load_image(f)
        abs_vec = compute_block_means(img)
        norm_vec = normalize_minmax(abs_vec)
        images.append(img)
        abs_vecs.append(abs_vec)
        norm_vecs.append(norm_vec)
        labels.append(-1)

    display_images_with_titles(images, [f"Label {l}" for l in labels])

    print("\n=== Абсолютні та нормовані вектори ===")
    for i, (abs_v, norm_v) in enumerate(zip(abs_vecs, norm_vecs)):
        print(f"Образ {i+1} (Label={labels[i]}):")
        print("Абсолютний:", np.array2string(abs_v, precision=2))
        print("Нормований:", np.array2string(norm_v, precision=3))

    X = np.array(norm_vecs)
    y = np.array(labels)

    
    perceptron = Perceptron(input_size=X.shape[1])
    perceptron.train(X, y)

   
    print("Виберіть невідомий образ для класифікації")
    unknown_files = choose_files(multiple=False)
    if not unknown_files:
        print("Невідомий образ не вибрано. Вихід.")
        return

    unknown_img = load_image(unknown_files[0])
    unknown_vec = normalize_minmax(compute_block_means(unknown_img))
    pred_class = perceptron.predict(unknown_vec)

    display_images_with_titles([unknown_img], [f"Невідомий\nLabel={pred_class}"])
    print(f"\nНевідомий образ належить до класу: {pred_class}")

    root = Tk()
    root.withdraw()
    messagebox.showinfo("Результат класифікації", f"Клас: {pred_class}")
    root.destroy()


if __name__ == "__main__":
    main()
