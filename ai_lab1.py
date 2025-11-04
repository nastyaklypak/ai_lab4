import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt


ROWS = 5
COLS = 5
THRESHOLD = 128 


class FeatureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ЛР1 – Ознакові вектори ")

       
        self.btn_open = tk.Button(root, text="Відкрити BMP…", command=self.open_image)
        self.btn_open.pack(pady=5)

       
        self.canvas = tk.Canvas(root, width=200, height=200, bg="white", relief="solid", bd=1)
        self.canvas.pack(pady=5)

       
        self.text = tk.Text(root, width=80, height=20)
        self.text.pack(padx=10, pady=10)

        self.image = None
        self.tk_img = None

    def open_image(self):
        path = filedialog.askopenfilename(
            title="Оберіть BMP", filetypes=[("BMP files", "*.bmp"), ("Усі файли", "*.*")]
        )
        if not path:
            return

        try:
            # Завантаження та бінаризація
            img = Image.open(path).convert("L")

            arr = np.array(img)
            
            binary_mask = arr < THRESHOLD

         
            binary_image = binary_mask.astype(np.uint8)

            arr = binary_image

            self.image = arr

           
            img_resized = img.resize((200, 200))
            self.tk_img = ImageTk.PhotoImage(img_resized)
            self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

            # Обчислити вектори
            self.compute_features()

        except Exception as e:
            messagebox.showerror("Помилка", str(e))





    def compute_features(self):
        if self.image is None:
            return

        h, w = self.image.shape
        vec = []

        row_edges = np.linspace(0, h, ROWS + 1, dtype=int)
        col_edges = np.linspace(0, w, COLS + 1, dtype=int)

        for r in range(ROWS):
            for c in range(COLS):
                r0, r1 = row_edges[r], row_edges[r + 1]
                c0, c1 = col_edges[c], col_edges[c + 1]
                block = self.image[r0:r1, c0:c1]
                cnt = int(block.sum())
                vec.append(cnt)

        abs_vec = vec
        s = sum(abs_vec)
        if s > 0:
            norm_vec = [v / s for v in abs_vec]
        else:
            norm_vec = [0 for _ in abs_vec]

       
        self.text.delete("1.0", tk.END)
        self.text.insert(tk.END, f"Абсолютний вектор (N={len(abs_vec)}):\n{abs_vec}\n\n")
        self.text.insert(tk.END, "Нормований вектор (за сумою):\n")
        self.text.insert(tk.END, "[" + ", ".join(f"{x:.3f}" for x in norm_vec) + "]\n\n")

        self.text.insert(tk.END, "Пояснення:\n")
        self.text.insert(tk.END, "- Абсолютний вектор: кількість чорних пікселів у кожному осередку сітки.\n")
        self.text.insert(tk.END, "- Нормований вектор: кожна координата поділена на суму всіх координат.\n")
        self.text.insert(tk.END, "- Щоб отримати більш \"розподілений\" вектор, можна збільшити розмір символу\n")
        self.text.insert(tk.END, "  на зображенні або збільшити розмір сітки (ROWS, COLS).\n")

    
        plt.figure(figsize=(6, 2))
        plt.bar(range(len(norm_vec)), norm_vec)
        plt.title("Normalized Feature Vector")
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.show()


if __name__ == "__main__":
    root = tk.Tk()
    app = FeatureApp(root)
    root.mainloop()
