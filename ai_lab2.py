import numpy as np
import cv2
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog, simpledialog, messagebox


RESIZE = (64, 64)   
BLOCKS = (8, 8)     


def load_image(path, gray=True): #винести в окремий метод !
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

def chebyshev_distance(a, b):
    return np.max(np.abs(a - b))

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

def main():
    # Вибрати еталони
    template_files = choose_files(multiple=True, title="Виберіть файли еталонів (3-4 шт.)")
    if not template_files or len(template_files) < 1:
        print("Еталони не вибрано. Вихід.")
        return
    #  до 4
    if len(template_files) > 4:
        template_files = template_files[:4]

    # Завантажити еталони та обчислити ознаки
    templates = []
    for p in template_files:
        img = load_image(p, gray=True)
        abs_vec = compute_block_means(img)
        norm_vec = normalize_minmax(abs_vec)
        templates.append({
            'path': p,
            'image': img,
            'abs': abs_vec,
            'norm': norm_vec
        })


    display_images_with_titles([t['image'] for t in templates],
                               [f"Еталон {i+1}\n{t['path'].split('/')[-1]}" for i,t in enumerate(templates)])

    # Показати абсолютні та нормовані вектори у консолі
    for i, t in enumerate(templates):
        print(f"\n=== Еталон {i+1}: {t['path']} ===")
        print("Абсолютний вектор (довжина={}):".format(len(t['abs'])))
        print(np.array2string(t['abs'], precision=2, separator=', '))
        print("Нормований вектор (мінмакс -> [0,1]):")
        print(np.array2string(t['norm'], precision=3, separator=', '))

    # Вибір невідомого образу
    unknown_files = choose_files(multiple=False, title="Виберіть файл невідомого образу")
    if not unknown_files:
        print("Невідомий образ не вибрано. Вихід.")
        return
    unknown_path = unknown_files[0]
    unknown_img = load_image(unknown_path, gray=True)
    unknown_abs = compute_block_means(unknown_img)
    unknown_norm = normalize_minmax(unknown_abs)

   
    display_images_with_titles([unknown_img], [f"Невідомий\n{unknown_path.split('/')[-1]}"])

    print("\n=== Невідомий образ ===")
    print("Абсолютний вектор:")
    print(np.array2string(unknown_abs, precision=2, separator=', '))
    print("Нормований вектор:")
    print(np.array2string(unknown_norm, precision=3, separator=', '))

    # Обчислити відстані Чебишева до кожного еталону (по нормованих векторах)
    distances = []
    for i, t in enumerate(templates):
        d = chebyshev_distance(unknown_norm, t['norm'])
        distances.append(d)
        print(f"Відстань Чебишева до Еталону {i+1}: {d:.4f}")

    # Класифікація: мінімальна відстань
    best_idx = int(np.argmin(distances))
    print(f"\nКласифікація: Невідомий образ належить до Еталону {best_idx+1} ({templates[best_idx]['path']})")
  
    root = Tk()
    root.withdraw()
    messagebox.showinfo("Результат класифікації",
                        f"Найменша відстань: {distances[best_idx]:.4f}\n"
                        f"Клас: Еталон {best_idx+1}\nФайл: {templates[best_idx]['path'].split('/')[-1]}")
    root.destroy()


    plt.figure(figsize=(6,4))
    idxs = np.arange(1, len(distances)+1)
    plt.bar(idxs, distances)
    plt.xticks(idxs, [f"Еталон {i}" for i in idxs])
    plt.ylabel("Відстань Чебишева")
    plt.title("Відстані невідомого образу до еталонів (нормовані вектори)")
    plt.show()

if __name__ == "__main__":
    main()
