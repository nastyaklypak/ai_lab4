import tkinter as tk
from tkinter import filedialog
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os


ROWS = 5          
COLS = 5          
THRESHOLD = 128   


classes_norm = {} 
classes_img = {}  
centroids = {}   


COLORS = {"A": "red", "B": "blue", "C": "green"}


def compute_features_and_normalize(path):
    
    img = Image.open(path).convert("L")
    img = img.resize((200, 200), Image.Resampling.LANCZOS)
    arr = np.array(img)
    
    # Бінаризація:  (темні) - 1,  (світлий) - 0
    binary_img = (arr < THRESHOLD).astype(np.uint8)
    
    h, w = binary_img.shape
    abs_vec = []
    
    row_edges = np.linspace(0, h, ROWS + 1, dtype=int)
    col_edges = np.linspace(0, w, COLS + 1, dtype=int)
    
   
    for r in range(ROWS):
        for c in range(COLS):
            block = binary_img[row_edges[r]:row_edges[r+1], 
                              col_edges[c]:col_edges[c+1]]
            abs_vec.append(int(block.sum()))
    
    abs_vec = np.array(abs_vec, dtype=np.float32)
    
    # Нормування
    total = np.sum(abs_vec)
    if total == 0:
        return np.zeros_like(abs_vec), binary_img, 0
    
    norm_vec = abs_vec / total
    
    return norm_vec, binary_img, abs_vec.sum()


def euclidean_distance(vec1, vec2):

    return np.linalg.norm(vec1 - vec2)


def add_sample(class_name, path):

    
    norm_vec, binary_img, abs_sum = compute_features_and_normalize(path)
    
    if class_name not in classes_norm:
        classes_norm[class_name] = []
        classes_img[class_name] = []
    

    classes_norm[class_name].append(norm_vec)
    classes_img[class_name].append(binary_img)
    
    # Геометричний центр
    norm_data = np.array(classes_norm[class_name])
    centroids[class_name] = np.mean(norm_data, axis=0)
    
    print(f"    Додано зразок: {os.path.basename(path)}")
    print(f"    Абсолютний (Сума): {abs_sum:.0f}, Нормований (Сума): {np.sum(norm_vec):.3f}")


def classify_by_centroid(unk_norm_vec):
  
    
    if not centroids:
        print(" Немає еталонів для класифікації!")
        return "Невідомий", 0
    
    min_dist = float('inf')
    final_class = "Невідомий"
    
    print("КЛАСИФІКАЦІЯ: МЕТОД ПОРІВНЯННЯ З ЕТАЛОНОМ")

    for class_name, center in sorted(centroids.items()):
        dist = euclidean_distance(unk_norm_vec, center)
        
        print(f"   Клас {class_name}: Евклідова відстань: {dist:.6f}")
        
        if dist < min_dist:
            min_dist = dist
            final_class = class_name
    
    print(f"\n фінальний результат: Клас {final_class} (Відстань: {min_dist:.6f})")
    return final_class, min_dist


def select_files(class_name):

    root = tk.Tk()
    root.withdraw()
    files = filedialog.askopenfilenames(
        title=f"Виберіть 10 файлів для класу {class_name}",
        filetypes=[("Зображення", "*.bmp *.png *.jpg")]
    )
    root.destroy()
    return list(files)


def select_file():

    root = tk.Tk()
    root.withdraw()
    file = filedialog.askopenfilename(
        title="Виберіть невідомий образ",
        filetypes=[("Зображення", "*.bmp *.png *.jpg")]
    )
    root.destroy()
    return file


def show_stats():

    print("поточні статичні параматри (геометричні цетри)")
    
    for cls in sorted(centroids.keys()):
        center = centroids[cls]
        print(f"Клас {cls}:")
        print(f"  Перші 5 нормованих ознак: {center[:5]}")
        print(f"  Сума елементів: {center.sum():.6f}")

def show_images():
  
    all_imgs = []
    titles = []
    
    for cls in sorted(classes_img.keys()):
        for i, img in enumerate(classes_img[cls], 1):
            all_imgs.append(img)
            titles.append(f"{cls}-{i}")
    
    n = len(all_imgs)
    if n == 0: return
    
    cols = 10 
    rows = (n + cols - 1) // cols
    
    plt.figure(figsize=(20, 2 * rows))
    for i in range(n):
        plt.subplot(rows, cols, i+1)

        plt.imshow(all_imgs[i], cmap='gray_r') 
        plt.title(titles[i], fontsize=8)
        plt.axis('off')
    plt.suptitle("Навчальні зразки", fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def show_clusters(unk_norm=None, result_cls=None):
  
    plt.figure(figsize=(10, 7))
    

    feature1_idx = 0 
    feature2_idx = 1 
    
    for cls in sorted(classes_norm.keys()):
        data = np.array(classes_norm[cls])
        color = COLORS[cls]
        

        plt.scatter(data[:, feature1_idx], data[:, feature2_idx], 
                   color=color, s=80, alpha=0.6, label=f"Клас {cls} (Зразки)", marker='o')
        
   
        center = centroids[cls]
        plt.scatter(center[feature1_idx], center[feature2_idx], 
                   marker="*", s=500, color=color, 
                   edgecolor="black", linewidth=2,
                   label=f"Клас {cls} (ЦЕНТР)", zorder=10)
    

    if unk_norm is not None:
        plt.scatter(unk_norm[feature1_idx], unk_norm[feature2_idx], 
                   marker="X", s=300, color="black", 
                   edgecolor="yellow", linewidth=3, 
                   label=f"НЕВІДОМИЙ → {result_cls}", zorder=15)
    
    plt.xlabel(f"Ознака {feature1_idx + 1} (Нормоване значення)")
    plt.ylabel(f"Ознака {feature2_idx + 1} (Нормоване значення)")
    plt.title("Проекція кластерів (Метод порівняння з еталоном)", 
              fontsize=12, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    
    class_names = ["A", "B", "C"]
    
    
    for cls in class_names:
        files = select_files(cls) 
        if not files:
            continue
        print(f"\nКлас {cls}: ({len(files)} зразків)")
        for f in files:
            add_sample(cls, f) 
    
    show_stats()

    show_images() 
    
    if not centroids:
        return
        


    unknown_file = select_file()
    
    if not unknown_file:
        return
    
    try:

        unk_norm, unk_img, unk_abs_sum = compute_features_and_normalize(unknown_file)
        

        plt.figure(figsize=(4, 4))
        plt.imshow(unk_img, cmap='gray_r')
        plt.title("Невідомий образ")
        plt.axis('off')
        plt.show()
        

        print(f"\nОзнакові значення невідомого образу (Крок 6):")
        print(f"  Абсолютний (Сума): {unk_abs_sum:.0f}")
        print(f"  Нормований (Перші 5): {unk_norm[:5]}")
        
 
        result, _ = classify_by_centroid(unk_norm)
        
        print(f"ФІНАЛЬНИЙ РЕЗУЛЬТАТ КЛАСИФІКАЦІЇ: Клас {result}")
     
        

        show_clusters(unk_norm, result) 
        
    except Exception as e:
        print(f"\n Критична помилка: {e}")

    print("\nПрограма завершена.")


if __name__ == "__main__":
    main()