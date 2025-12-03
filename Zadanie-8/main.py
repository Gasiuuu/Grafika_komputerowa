import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
from typing import Optional, Tuple
import re


class PPMLoader:
    @staticmethod
    def loadPPM(filepath: str) -> Tuple[np.ndarray, dict]:
        try:
            with open(filepath, 'rb') as f:
                magic = PPMLoader.readToken(f)

                if magic not in [b'P3', b'P6']:
                    raise ValueError(f"Nieobsługiwany format: {magic.decode('ascii', errors='ignore')}")

                width = int(PPMLoader.readToken(f))
                height = int(PPMLoader.readToken(f))
                maxval = int(PPMLoader.readToken(f))

                if maxval > 255:
                    raise ValueError(f"Nieobsługiwana maksymalna wartość koloru: {maxval}")

                metadata = {
                    'format': magic.decode('ascii'),
                    'width': width,
                    'height': height,
                    'maxval': maxval
                }

                total_pixels = width * height * 3

                if magic == b'P3':
                    data = f.read()
                    data_no_comments = re.sub(rb'#.*', b'', data)
                    tokens = data_no_comments.split()

                    if len(tokens) < total_pixels:
                        raise ValueError(
                            f"Nieprawidłowa liczba pikseli: oczekiwano {total_pixels}, otrzymano {len(tokens)}")

                    pixel_iter = (int(t) for t in tokens)
                    image_array = np.fromiter(pixel_iter, dtype=np.uint8, count=total_pixels)
                    image_array = image_array.reshape((height, width, 3))

                elif magic == b'P6':
                    total_bytes = width * height * 3
                    data = f.read(total_bytes)

                    if len(data) != total_bytes:
                        raise ValueError(
                            f"Nieprawidłowa liczba bajtów: oczekiwano {total_bytes}, otrzymano {len(data)}")

                    image_array = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))

                return image_array, metadata

        except FileNotFoundError:
            raise FileNotFoundError(f"Nie znaleziono pliku: {filepath}")
        except Exception as e:
            raise Exception(f"Błąd podczas wczytywania pliku PPM: {str(e)}")

    @staticmethod
    def readToken(f) -> bytes:
        token = b''
        while True:
            char = f.read(1)
            if not char:
                return token
            if char == b'#':
                while char and char != b'\n':
                    char = f.read(1)
                continue
            if char in b' \t\n\r':
                if token:
                    return token
                continue
            token += char


class MorphologicalOperations:

    @staticmethod
    def toBinary(image: np.ndarray, threshold: int = 128) -> np.ndarray:
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2).astype(np.uint8)
        else:
            gray = image
        return (gray > threshold).astype(np.uint8) * 255

    @staticmethod
    def dilation(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        binary = (image > 128).astype(np.uint8)
        h, w = binary.shape[:2]
        kh, kw = kernel.shape
        pad_h, pad_w = kh // 2, kw // 2

        padded = np.pad(binary, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
        result = np.zeros_like(binary)

        for i in range(h):
            for j in range(w):
                region = padded[i:i + kh, j:j + kw]
                if np.any(region * kernel):
                    result[i, j] = 1

        return result * 255

    @staticmethod
    def erosion(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        binary = (image > 128).astype(np.uint8)
        h, w = binary.shape[:2]
        kh, kw = kernel.shape
        pad_h, pad_w = kh // 2, kw // 2

        padded = np.pad(binary, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
        result = np.zeros_like(binary)

        for i in range(h):
            for j in range(w):
                region = padded[i:i + kh, j:j + kw]
                if np.all(region[kernel == 1] == 1):
                    result[i, j] = 1

        return result * 255

    @staticmethod
    def opening(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        eroded = MorphologicalOperations.erosion(image, kernel)
        return MorphologicalOperations.dilation(eroded, kernel)

    @staticmethod
    def closing(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        dilated = MorphologicalOperations.dilation(image, kernel)
        return MorphologicalOperations.erosion(dilated, kernel)

    @staticmethod
    def hitOrMiss(image: np.ndarray, kernel_hit: np.ndarray, kernel_miss: np.ndarray) -> np.ndarray:
        binary = (image > 128).astype(np.uint8)
        h, w = binary.shape[:2]
        kh, kw = kernel_hit.shape
        pad_h, pad_w = kh // 2, kw // 2

        padded = np.pad(binary, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
        result = np.zeros_like(binary)

        for i in range(h):
            for j in range(w):
                region = padded[i:i + kh, j:j + kw]

                hit_match = True
                for ki in range(kh):
                    for kj in range(kw):
                        if kernel_hit[ki, kj] == 1 and region[ki, kj] != 1:
                            hit_match = False
                            break
                    if not hit_match:
                        break

                miss_match = True
                for ki in range(kh):
                    for kj in range(kw):
                        if kernel_miss[ki, kj] == 1 and region[ki, kj] != 0:
                            miss_match = False
                            break
                    if not miss_match:
                        break

                if hit_match and miss_match:
                    result[i, j] = 1

        return result * 255

    @staticmethod
    def thinning(image: np.ndarray, iterations: int = 1) -> np.ndarray:
        kernels_hit = [
            np.array([[0, 0, 0], [1, 1, 0], [1, 1, 0]]),
            np.array([[1, 0, 0], [1, 1, 0], [1, 0, 0]]),
            np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]]),
            np.array([[1, 1, 1], [0, 1, 0], [0, 0, 0]]),
        ]

        kernels_miss = [
            np.array([[1, 1, 1], [0, 0, 1], [0, 0, 1]]),
            np.array([[0, 1, 1], [0, 0, 1], [0, 1, 1]]),
            np.array([[0, 0, 1], [0, 0, 1], [1, 1, 1]]),
            np.array([[0, 0, 0], [1, 0, 1], [1, 1, 1]]),
        ]

        result = image.copy()
        for _ in range(iterations):
            for hit, miss in zip(kernels_hit, kernels_miss):
                hom = MorphologicalOperations.hitOrMiss(result, hit, miss)
                result = result - hom
                result = np.clip(result, 0, 255).astype(np.uint8)

        return result

    @staticmethod
    def thickening(image: np.ndarray, iterations: int = 1) -> np.ndarray:
        kernels_hit = [
            np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]]),
            np.array([[1, 0, 0], [1, 1, 0], [1, 0, 0]]),
            np.array([[0, 0, 0], [1, 1, 0], [1, 1, 0]]),
            np.array([[0, 0, 0], [0, 1, 0], [1, 1, 1]]),
        ]

        kernels_miss = [
            np.array([[0, 0, 1], [0, 0, 1], [1, 1, 1]]),
            np.array([[0, 1, 1], [0, 0, 1], [0, 1, 1]]),
            np.array([[1, 1, 1], [0, 0, 1], [0, 0, 1]]),
            np.array([[1, 1, 1], [1, 0, 1], [0, 0, 0]]),
        ]

        result = (image > 128).astype(np.uint8) * 255

        for _ in range(iterations):
            for hit, miss in zip(kernels_hit, kernels_miss):
                hom = MorphologicalOperations.hitOrMiss(result, hit, miss)
                result = result | hom

        return result.astype(np.uint8)


class StructuringElementDialog(tk.Toplevel):

    def __init__(self, parent):
        super().__init__(parent)
        self.title("Definiuj Element Strukturyzujący")
        self.geometry("400x350")
        self.transient(parent)
        self.grab_set()

        self.result = None
        self.size = 3
        self.buttons = []

        self.setupUI()

    def setupUI(self):
        self.grid_frame = ttk.Frame(self)
        self.grid_frame.pack(pady=10, padx=10, expand=True, fill=tk.BOTH)

        preset_frame = ttk.LabelFrame(self, text="Predefiniowane kształty")
        preset_frame.pack(pady=10, padx=10, fill=tk.X)

        ttk.Button(preset_frame, text="Krzyż", command=self.setCross).pack(side=tk.LEFT, padx=5)
        ttk.Button(preset_frame, text="Kwadrat", command=self.setSquare).pack(side=tk.LEFT, padx=5)
        ttk.Button(preset_frame, text="Pozioma linia", command=self.setHLine).pack(side=tk.LEFT, padx=5)
        ttk.Button(preset_frame, text="Pionowa linia", command=self.setVLine).pack(side=tk.LEFT, padx=5)

        button_frame = ttk.Frame(self)
        button_frame.pack(pady=10)

        ttk.Button(button_frame, text="OK", command=self.onOK).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Anuluj", command=self.onCancel).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Wyczyść", command=self.clear).pack(side=tk.LEFT, padx=5)

        self.createGrid()

    def createGrid(self):
        for widget in self.grid_frame.winfo_children():
            widget.destroy()

        self.buttons = []
        for i in range(3):
            row = []
            for j in range(3):
                btn = tk.Button(self.grid_frame, text="0", width=3, height=1,
                                bg="white", command=lambda r=i, c=j: self.toggleButton(r, c))
                btn.grid(row=i, column=j, padx=2, pady=2)
                row.append(btn)
            self.buttons.append(row)

    def toggleButton(self, row, col):
        btn = self.buttons[row][col]
        if btn['text'] == '0':
            btn.config(text='1', bg='black', fg='white')
        else:
            btn.config(text='0', bg='white', fg='black')

    def getKernel(self) -> np.ndarray:
        kernel = np.zeros((3, 3), dtype=np.uint8)
        for i in range(3):
            for j in range(3):
                kernel[i, j] = 1 if self.buttons[i][j]['text'] == '1' else 0
        return kernel

    def setCross(self):
        self.clear()
        self.buttons[1][0].config(text='1', bg='black', fg='white')
        self.buttons[1][1].config(text='1', bg='black', fg='white')
        self.buttons[1][2].config(text='1', bg='black', fg='white')
        self.buttons[0][1].config(text='1', bg='black', fg='white')
        self.buttons[2][1].config(text='1', bg='black', fg='white')

    def setSquare(self):
        for i in range(3):
            for j in range(3):
                self.buttons[i][j].config(text='1', bg='black', fg='white')

    def setHLine(self):
        self.clear()
        for j in range(3):
            self.buttons[1][j].config(text='1', bg='black', fg='white')

    def setVLine(self):
        self.clear()
        for i in range(3):
            self.buttons[i][1].config(text='1', bg='black', fg='white')

    def clear(self):
        for i in range(3):
            for j in range(3):
                self.buttons[i][j].config(text='0', bg='white', fg='black')

    def onOK(self):
        self.result = self.getKernel()
        self.destroy()

    def onCancel(self):
        self.result = None
        self.destroy()


class ImageViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Operatory Morfologiczne - Przeglądarka Obrazów")
        self.root.geometry("1400x900")

        self.original_image: Optional[np.ndarray] = None
        self.processed_image: Optional[np.ndarray] = None
        self.display_image: Optional[np.ndarray] = None
        self.photo_image: Optional[ImageTk.PhotoImage] = None
        self.metadata: dict = {}
        self.current_filepath: Optional[str] = None

        self.structuring_element = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8)

        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0

        self.is_panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0

        self.setupUI()

    def setupUI(self):
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        ttk.Button(control_frame, text="Otwórz obraz", command=self.loadImage).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Zapisz JPEG", command=self.saveAsJpeg).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Reset", command=self.resetImage).pack(side=tk.LEFT, padx=5)

        ttk.Label(control_frame, text="Powiększenie:").pack(side=tk.LEFT, padx=(20, 5))
        ttk.Button(control_frame, text="-", command=self.zoomOut, width=3).pack(side=tk.LEFT)
        self.zoom_label = ttk.Label(control_frame, text="100%")
        self.zoom_label.pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="+", command=self.zoomIn, width=3).pack(side=tk.LEFT)
        ttk.Button(control_frame, text="Reset widoku", command=self.resetView).pack(side=tk.LEFT, padx=5)

        morph_frame = ttk.LabelFrame(self.root, text="Operacje Morfologiczne")
        morph_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        left_col = ttk.Frame(morph_frame)
        left_col.pack(side=tk.LEFT, padx=5, pady=5)

        ttk.Button(left_col, text="Definiuj element strukturyzujący",
                   command=self.defineStructuringElement).pack(pady=2, fill=tk.X)
        ttk.Button(left_col, text="Dylatacja", command=self.applyDilation).pack(pady=2, fill=tk.X)
        ttk.Button(left_col, text="Erozja", command=self.applyErosion).pack(pady=2, fill=tk.X)

        mid_col = ttk.Frame(morph_frame)
        mid_col.pack(side=tk.LEFT, padx=5, pady=5)

        ttk.Button(mid_col, text="Otwarcie", command=self.applyOpening).pack(pady=2, fill=tk.X)
        ttk.Button(mid_col, text="Domknięcie", command=self.applyClosing).pack(pady=2, fill=tk.X)
        ttk.Button(mid_col, text="Hit-or-Miss", command=self.applyHitOrMiss).pack(pady=2, fill=tk.X)

        right_col = ttk.Frame(morph_frame)
        right_col.pack(side=tk.LEFT, padx=5, pady=5)

        thin_frame = ttk.Frame(right_col)
        thin_frame.pack(pady=2, fill=tk.X)
        ttk.Button(thin_frame, text="Pocienianie", command=self.applyThinning).pack(side=tk.LEFT)
        # self.thin_iter = tk.IntVar(value=1)
        # ttk.Spinbox(thin_frame, from_=1, to=10, width=5, textvariable=self.thin_iter).pack(side=tk.LEFT, padx=5)

        thick_frame = ttk.Frame(right_col)
        thick_frame.pack(pady=2, fill=tk.X)
        ttk.Button(thick_frame, text="Pogrubianie", command=self.applyThickening).pack(side=tk.LEFT)
        # self.thick_iter = tk.IntVar(value=1)
        # ttk.Spinbox(thick_frame, from_=1, to=10, width=5, textvariable=self.thick_iter).pack(side=tk.LEFT, padx=5)

        info_frame = ttk.Frame(self.root)
        info_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        self.info_label = ttk.Label(info_frame, text="Nie załadowano obrazu", relief=tk.SUNKEN)
        self.info_label.pack(fill=tk.X)

        canvas_frame = ttk.Frame(self.root)
        canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.canvas = tk.Canvas(canvas_frame, bg='gray')
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar = ttk.Scrollbar(self.root, orient=tk.HORIZONTAL, command=self.canvas.xview)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X, padx=5)

        self.canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)

        self.canvas.bind("<ButtonPress-1>", self.onMousePress)
        self.canvas.bind("<B1-Motion>", self.onMouseDrag)
        self.canvas.bind("<ButtonRelease-1>", self.onMouseRelease)
        self.canvas.bind("<Motion>", self.onMouseMove)
        self.canvas.bind("<MouseWheel>", self.onMousewheel)

    def defineStructuringElement(self):
        dialog = StructuringElementDialog(self.root)
        self.root.wait_window(dialog)
        if dialog.result is not None:
            self.structuring_element = dialog.result
            messagebox.showinfo("Sukces", "Element strukturyzujący został zdefiniowany")

    def applyDilation(self):
        if self.original_image is None:
            messagebox.showwarning("Uwaga", "Najpierw wczytaj obraz!")
            return

        binary = MorphologicalOperations.toBinary(self.original_image)
        result = MorphologicalOperations.dilation(binary, self.structuring_element)
        self.processed_image = np.stack([result, result, result], axis=2)
        self.updateDisplay()

    def applyErosion(self):
        if self.original_image is None:
            messagebox.showwarning("Uwaga", "Najpierw wczytaj obraz!")
            return

        binary = MorphologicalOperations.toBinary(self.original_image)
        result = MorphologicalOperations.erosion(binary, self.structuring_element)
        self.processed_image = np.stack([result, result, result], axis=2)
        self.updateDisplay()

    def applyOpening(self):
        if self.original_image is None:
            messagebox.showwarning("Uwaga", "Najpierw wczytaj obraz!")
            return

        binary = MorphologicalOperations.toBinary(self.original_image)
        result = MorphologicalOperations.opening(binary, self.structuring_element)
        self.processed_image = np.stack([result, result, result], axis=2)
        self.updateDisplay()

    def applyClosing(self):
        if self.original_image is None:
            messagebox.showwarning("Uwaga", "Najpierw wczytaj obraz!")
            return

        binary = MorphologicalOperations.toBinary(self.original_image)
        result = MorphologicalOperations.closing(binary, self.structuring_element)
        self.processed_image = np.stack([result, result, result], axis=2)
        self.updateDisplay()

    def applyHitOrMiss(self):
        if self.original_image is None:
            messagebox.showwarning("Uwaga", "Najpierw wczytaj obraz!")
            return

        binary = MorphologicalOperations.toBinary(self.original_image)

        kernel_hit = self.structuring_element
        kernel_miss = 1 - self.structuring_element

        result = MorphologicalOperations.hitOrMiss(binary, kernel_hit, kernel_miss)
        self.processed_image = np.stack([result, result, result], axis=2)
        self.updateDisplay()

    def applyThinning(self):
        if self.original_image is None:
            messagebox.showwarning("Uwaga", "Najpierw wczytaj obraz!")
            return

        binary = MorphologicalOperations.toBinary(self.original_image)
        # iterations = self.thin_iter.get()
        result = MorphologicalOperations.thinning(binary, 1)
        self.processed_image = np.stack([result, result, result], axis=2)
        self.updateDisplay()

    def applyThickening(self):
        if self.original_image is None:
            messagebox.showwarning("Uwaga", "Najpierw wczytaj obraz!")
            return

        binary = MorphologicalOperations.toBinary(self.original_image)
        # iterations = self.thick_iter.get()
        result = MorphologicalOperations.thickening(binary, 1)
        self.processed_image = np.stack([result, result, result], axis=2)
        self.updateDisplay()

    def resetImage(self):
        if self.original_image is None:
            messagebox.showwarning("Uwaga", "Najpierw wczytaj obraz!")
            return

        self.processed_image = None
        self.updateDisplay()

    def loadImage(self):
        filetypes = [
            ("Wszystkie obsługiwane", "*.ppm *.jpg *.jpeg"),
            ("Pliki PPM", "*.ppm"),
            ("Pliki JPEG", "*.jpg *.jpeg"),
            ("Wszystkie pliki", "*.*")
        ]

        filepath = filedialog.askopenfilename(title="Wybierz obraz", filetypes=filetypes)

        if not filepath:
            return

        try:
            ext = filepath.lower().split('.')[-1]

            if ext == 'ppm':
                self.original_image, self.metadata = PPMLoader.loadPPM(filepath)
                self.current_filepath = filepath

            elif ext in ['jpg', 'jpeg']:
                img = Image.open(filepath)
                self.original_image = np.array(img)
                self.current_filepath = filepath
                self.metadata = {
                    'format': 'JPEG',
                    'width': img.width,
                    'height': img.height,
                    'maxval': 255
                }

            self.processed_image = None
            self.resetView()
            self.updateDisplay()
            self.updateInfoLabel()

        except Exception as e:
            messagebox.showerror("Błąd", f"Nie udało się wczytać obrazu:\n{str(e)}")

    def saveAsJpeg(self):
        if self.display_image is None:
            messagebox.showwarning("Uwaga", "Brak obrazu do zapisania!")
            return

        filepath = filedialog.asksaveasfilename(
            title="Zapisz jako JPEG",
            defaultextension=".jpg",
            filetypes=[("Pliki JPEG", "*.jpg"), ("Wszystkie pliki", "*.*")]
        )

        if not filepath:
            return

        try:
            if len(self.display_image.shape) == 2:
                img = Image.fromarray(self.display_image, mode='L')
            else:
                img = Image.fromarray(self.display_image, mode='RGB')

            img.save(filepath, 'JPEG', quality=95)
            messagebox.showinfo("Sukces", f"Obraz zapisany jako: {filepath}")

        except Exception as e:
            messagebox.showerror("Błąd", f"Nie udało się zapisać obrazu:\n{str(e)}")

    def updateDisplay(self):
        if self.original_image is None:
            return

        if self.processed_image is not None:
            self.display_image = self.processed_image.copy()
        else:
            self.display_image = self.original_image.copy()

        if self.zoom_level != 1.0:
            h, w = self.display_image.shape[:2]
            new_h = int(h * self.zoom_level)
            new_w = int(w * self.zoom_level)

            if len(self.display_image.shape) == 2:
                img = Image.fromarray(self.display_image, mode='L')
            else:
                img = Image.fromarray(self.display_image, mode='RGB')

            img = img.resize((new_w, new_h), Image.NEAREST)
            display_array = np.array(img)
        else:
            display_array = self.display_image

        if len(display_array.shape) == 2:
            img = Image.fromarray(display_array, mode='L')
        else:
            img = Image.fromarray(display_array, mode='RGB')

        self.photo_image = ImageTk.PhotoImage(img)

        self.canvas.delete("all")
        self.canvas.create_image(self.pan_x, self.pan_y, anchor=tk.NW, image=self.photo_image)

        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def updateInfoLabel(self):
        if self.original_image is None:
            self.info_label.config(text="Nie załadowano obrazu")
            return

        info_text = (
            f"Format: {self.metadata.get('format', 'N/A')} | "
            f"Wymiary: {self.metadata.get('width', 0)}x{self.metadata.get('height', 0)} | "
            f"Max wartość: {self.metadata.get('maxval', 0)} | "
            f"Plik: {self.current_filepath if self.current_filepath else 'N/A'}"
        )
        self.info_label.config(text=info_text)

    def zoomIn(self):
        self.zoom_level = min(self.zoom_level * 1.2, 10.0)
        self.zoom_label.config(text=f"{int(self.zoom_level * 100)}%")
        self.updateDisplay()

    def zoomOut(self):
        self.zoom_level = max(self.zoom_level / 1.2, 0.1)
        self.zoom_label.config(text=f"{int(self.zoom_level * 100)}%")
        self.updateDisplay()

    def resetView(self):
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.zoom_label.config(text="100%")
        self.updateDisplay()

    def onMousePress(self, event):
        self.is_panning = True
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        self.canvas.config(cursor="fleur")

    def onMouseDrag(self, event):
        if self.is_panning:
            dx = event.x - self.pan_start_x
            dy = event.y - self.pan_start_y
            self.pan_x += dx
            self.pan_y += dy
            self.pan_start_x = event.x
            self.pan_start_y = event.y
            self.updateDisplay()

    def onMouseRelease(self, event):
        self.is_panning = False
        self.canvas.config(cursor="")

    def onMouseMove(self, event):
        if self.display_image is None:
            return

        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        img_x = int((canvas_x - self.pan_x) / self.zoom_level)
        img_y = int((canvas_y - self.pan_y) / self.zoom_level)

        if self.original_image is not None:
            h, w = self.original_image.shape[:2]
            if 0 <= img_x < w and 0 <= img_y < h:
                if len(self.original_image.shape) == 2:
                    pixel_value = self.original_image[img_y, img_x]
                    pixel_info = f"Pozycja: ({img_x}, {img_y}) | Wartość: {pixel_value}"
                else:
                    r, g, b = self.original_image[img_y, img_x]
                    pixel_info = f"Pozycja: ({img_x}, {img_y}) | RGB: ({r}, {g}, {b})"

                self.info_label.config(text=pixel_info)

    def onMousewheel(self, event):
        if event.delta > 0:
            self.zoomIn()
        else:
            self.zoomOut()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageViewer(root)
    root.mainloop()

