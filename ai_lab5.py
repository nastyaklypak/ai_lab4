import numpy as np
import cv2
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog, messagebox

RESIZE = (64, 64)   
BLOCKS = (16, 16)     
BIN_THRESHOLD = 0.5   
MAX_ITER = 100       
SHOW_MATRICES = True 

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

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Не можу відкрити " + path)
    return img.astype(np.float32)

def compute_block_means(img, resize=RESIZE, blocks=BLOCKS):
    img_small = cv2.resize(img, resize, interpolation=cv2.INTER_AREA)
    h, w = resize
    bh = h // blocks[0]
    bw = w // blocks[1]
    feats = []
    for i in range(blocks[0]):
        for j in range(blocks[1]):
            y0, y1 = i*bh, (i+1)*bh
            x0, x1 = j*bw, (j+1)*bw
            block = img_small[y0:y1, x0:x1]
            feats.append(block.mean())
    return np.array(feats, dtype=np.float32)

def normalize_minmax(vec):
    vmin = vec.min()
    vmax = vec.max()
    if np.isclose(vmax, vmin):
        return np.zeros_like(vec)
    return (vec - vmin) / (vmax - vmin)

def to_bipolar(vec, threshold=BIN_THRESHOLD):
    # додаємо невеликий шум для різних образів
    vec_noisy = vec + np.random.uniform(-0.01, 0.01, vec.shape)
    return np.where(vec_noisy >= threshold, 1, -1).astype(np.int8)

def hebbian_weights(patterns):
    m, n = patterns.shape
    W = np.zeros((n, n), dtype=np.float32)
    for p in patterns:
        W += np.outer(p, p)
    np.fill_diagonal(W, 0.0)
    return W / m

def hopfield_recall(W, init_state, max_iter=MAX_ITER, synchronous=True):
    state = init_state.copy()
    history = [state.copy()]
    n = state.size
    for it in range(max_iter):
        if synchronous:
            net = W.dot(state)
            new_state = np.where(net >= 0, 1, -1)
        else:
            new_state = state.copy()
            idxs = np.random.permutation(n)
            for idx in idxs:
                net_i = W[idx].dot(new_state)
                new_state[idx] = 1 if net_i >= 0 else -1
        history.append(new_state.copy())
        if np.array_equal(new_state, state):
            return new_state, history
        state = new_state
    return state, history

def vector_to_block_image(vec, blocks=BLOCKS, resize=RESIZE):
    block = vec.reshape(blocks)
    img_blocks = ((block + 1) / 2.0 * 255).astype(np.uint8)
    bh = resize[0] // blocks[0]
    bw = resize[1] // blocks[1]
    img = np.kron(img_blocks, np.ones((bh, bw), dtype=np.uint8))
    return img

def show_weight_matrix(W):
    plt.figure(figsize=(5,5))
    plt.imshow(W, cmap='seismic', interpolation='nearest')
    plt.title("Матриця W")
    plt.colorbar()
    plt.show()

def show_images(images, titles):
    n = len(images)
    plt.figure(figsize=(4*n, 3))
    for i, img in enumerate(images):
        plt.subplot(1, n, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    print("=== Початок. Завантажуємо зразки для 3 класів ===")
    all_files = []
    for cls in ['A', 'B', 'C']:
        print(f"Виберіть файли для класу {cls}")
        files = choose_files(multiple=True, title=f"Файли класу {cls}")
        if len(files) < 1:
            print("Не вибрано файлів — вихід.")
            return
        all_files.append(files)

    orig_imgs, abs_vecs, norm_vecs = [], [], []
    for group in all_files:
        for f in group:
            img = load_image(f)
            orig_imgs.append(img)
            abs_v = compute_block_means(img)
            abs_vecs.append(abs_v)
            norm_vecs.append(normalize_minmax(abs_v))

    abs_vecs = np.array(abs_vecs)
    norm_vecs = np.array(norm_vecs)
    m, n = norm_vecs.shape
    print(f"Завантажено {m} зразків, розмір ознакового вектора = {n}")

    bipolar = np.array([to_bipolar(v) for v in norm_vecs])
    print("Біполярні вектори (перші 3):")
    for i in range(min(3, m)):
        print(bipolar[i])

    W = hebbian_weights(bipolar)
    print("Матриця ваг обчислена.")
    if SHOW_MATRICES:
        show_weight_matrix(W)

    cap = int(0.15 * n)
    print(f"Орієнтовна місткість = {cap} образів. Завантажено m={m}.")
    if m > cap:
        print("Увага: може бути нестійкість (перехресні асоціації).")
    else:
        print("Місткості достатньо.")

    restored = []
    for i, p in enumerate(bipolar):
        rec, hist = hopfield_recall(W, p, max_iter=MAX_ITER, synchronous=True)
        restored.append(rec)
        print(f"Еталон {i+1} відновився за {len(hist)-1} ітерацій")
    restored = np.array(restored)

    for i in range(min(4, len(orig_imgs))):
        img_orig = orig_imgs[i].astype(np.uint8)
        img_pattern = vector_to_block_image(bipolar[i])
        img_restored = vector_to_block_image(restored[i])
        show_images([img_orig, img_pattern, img_restored],
                    [f"Оригінал #{i+1}", "Еталон (-1/1)", "Відновлений"])

    print("Виберіть невідомий образ для класифікації")
    unk = choose_files(multiple=False, title="Невідомий образ")
    if not unk:
        print("Невідомий не вибрано. Кінець.")
        return
    uimg = load_image(unk[0])
    uabs = compute_block_means(uimg)
    unorm = normalize_minmax(uabs)
    ubip = to_bipolar(unorm)

    show_images([uimg.astype(np.uint8), vector_to_block_image(ubip)],
                ["Невідомий (оригінал)", "Невідомий (біполярний)"])

    result, hist = hopfield_recall(W, ubip, max_iter=MAX_ITER, synchronous=True)
    print(f"Відновлено за {len(hist)-1} ітерацій")

    show_images([uimg.astype(np.uint8), vector_to_block_image(result)],
                ["Невідомий (оригінал)", "Відновлений Хопфілд"])

    sims = [np.mean(p == result) for p in bipolar]
    sims = np.array(sims)
    best = np.argmax(sims)
    print(f"Найближчий еталон: #{best+1} (схожість {sims[best]*100:.1f}%)")
    root = Tk(); root.withdraw()
    messagebox.showinfo("Результат", f"Найближчий еталон: #{best+1}\nСхожість: {sims[best]*100:.1f}%")
    root.destroy()

if __name__ == "__main__":
    main()
