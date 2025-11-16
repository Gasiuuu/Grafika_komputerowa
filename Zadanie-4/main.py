import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image,  ImageTk
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
                    total_bytes = total_pixels
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


class ImageProcessor:

    @staticmethod
    def add_value(image: np.ndarray, value: float) -> np.ndarray:
        result = image.astype(np.float32) + value
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def subtract_value(image: np.ndarray, value: float) -> np.ndarray:
        result = image.astype(np.float32) - value
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def multiply_value(image: np.ndarray, value: float) -> np.ndarray:
        result = image.astype(np.float32) * value
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def divide_value(image: np.ndarray, value: float) -> np.ndarray:
        if value == 0:
            return image
        result = image.astype(np.float32) / value
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def change_brightness(image: np.ndarray, level: float) -> np.ndarray:
        result = image.astype(np.float32) + level
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def to_grayscale_average(image: np.ndarray) -> np.ndarray:
        gray = np.mean(image, axis=2, dtype=np.float32)
        return np.stack([gray, gray, gray], axis=2).astype(np.uint8)

    @staticmethod
    def to_grayscale_luminosity(image: np.ndarray) -> np.ndarray:
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        return np.stack([gray, gray, gray], axis=2).astype(np.uint8)

    @staticmethod
    def apply_convolution(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        h, w, c = image.shape
        kh, kw = kernel.shape
        pad_h, pad_w = kh // 2, kw // 2

        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='edge')
        result = np.zeros_like(image, dtype=np.float32)

        for i in range(h):
            for j in range(w):
                for k in range(c):
                    region = padded[i:i + kh, j:j + kw, k]
                    result[i, j, k] = np.sum(region * kernel)

        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def smoothing_filter(image: np.ndarray, size: int = 3) -> np.ndarray:
        kernel = np.ones((size, size), dtype=np.float32) / (size * size)
        return ImageProcessor.apply_convolution(image, kernel)

    @staticmethod
    def median_filter(image: np.ndarray, size: int = 3) -> np.ndarray:
        h, w, c = image.shape
        pad = size // 2
        padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
        result = np.zeros_like(image)

        for i in range(h):
            for j in range(w):
                for k in range(c):
                    region = padded[i:i + size, j:j + size, k]
                    result[i, j, k] = np.median(region)

        return result.astype(np.uint8)

    @staticmethod
    def sobel_filter(image: np.ndarray) -> np.ndarray:
        gray = np.mean(image, axis=2, dtype=np.float32)

        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=np.float32)

        sobel_y = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]], dtype=np.float32)

        h, w = gray.shape
        pad = 1
        padded = np.pad(gray, ((pad, pad), (pad, pad)), mode='edge')

        gx = np.zeros_like(gray)
        gy = np.zeros_like(gray)

        for i in range(h):
            for j in range(w):
                region = padded[i:i + 3, j:j + 3]
                gx[i, j] = np.sum(region * sobel_x)
                gy[i, j] = np.sum(region * sobel_y)

        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)

        return np.stack([magnitude, magnitude, magnitude], axis=2)

    @staticmethod
    def sharpening_filter(image: np.ndarray) -> np.ndarray:
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]], dtype=np.float32)
        return ImageProcessor.apply_convolution(image, kernel)

    @staticmethod
    def gaussian_blur(image: np.ndarray, size: int = 5, sigma: float = 1.0) -> np.ndarray:
        ax = np.arange(-size // 2 + 1., size // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
        kernel = kernel / np.sum(kernel)

        return ImageProcessor.apply_convolution(image, kernel)


class ImageViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Przeglądarka obrazów PPM/JPEG")
        self.root.geometry("1400x900")

        self.original_image: Optional[np.ndarray] = None
        self.processed_image: Optional[np.ndarray] = None
        self.display_image: Optional[np.ndarray] = None
        self.photo_image: Optional[ImageTk.PhotoImage] = None
        self.metadata: dict = {}
        self.current_filepath: Optional[str] = None

        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.color_scale = 1.0

        self.is_panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0

        self.setupUI()

    def setupUI(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Plik", menu=file_menu)
        file_menu.add_command(label="Otwórz obraz", command=self.loadImage)
        file_menu.add_command(label="Zapisz JPEG", command=self.saveAsJpeg)
        file_menu.add_separator()
        file_menu.add_command(label="Wyjście", command=self.root.quit)

        transform_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Przekształcenia punktowe", menu=transform_menu)
        transform_menu.add_command(label="Dodawanie", command=self.addValueDialog)
        transform_menu.add_command(label="Odejmowanie", command=self.subtractValueDialog)
        transform_menu.add_command(label="Mnożenie", command=self.multiplyValueDialog)
        transform_menu.add_command(label="Dzielenie", command=self.divideValueDialog)
        transform_menu.add_command(label="Zmiana jasności", command=self.brightnessDialog)
        transform_menu.add_separator()
        transform_menu.add_command(label="Skala szarości (średnia)", command=self.toGrayscaleAverage)
        transform_menu.add_command(label="Skala szarości (luminancja)", command=self.toGrayscaleLuminosity)

        filter_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Filtry", menu=filter_menu)
        filter_menu.add_command(label="Filtr wygładzający", command=self.applySmoothingFilter)
        filter_menu.add_command(label="Filtr medianowy", command=self.applyMedianFilter)
        filter_menu.add_command(label="Filtr Sobela (krawędzie)", command=self.applySobelFilter)
        filter_menu.add_command(label="Filtr wyostrzający", command=self.applySharpeningFilter)
        filter_menu.add_command(label="Rozmycie Gaussowskie", command=self.applyGaussianBlur)

        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Widok", menu=view_menu)
        view_menu.add_command(label="Reset obrazu", command=self.resetImage)
        view_menu.add_command(label="Reset widoku", command=self.resetView)

        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        ttk.Button(control_frame, text="Otwórz obraz", command=self.loadImage).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Reset obrazu", command=self.resetImage).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Zapisz JPEG", command=self.saveAsJpeg).pack(side=tk.LEFT, padx=5)

        ttk.Label(control_frame, text="Skalowanie kolorów:").pack(side=tk.LEFT, padx=(20, 5))
        self.color_scale_var = tk.DoubleVar(value=1.0)
        color_scale_slider = ttk.Scale(control_frame, from_=0.1, to=3.0, orient=tk.HORIZONTAL,
                                       variable=self.color_scale_var, command=self.onColorScaleChange,
                                       length=200)
        color_scale_slider.pack(side=tk.LEFT, padx=5)
        self.color_scale_label = ttk.Label(control_frame, text="1.0x")
        self.color_scale_label.pack(side=tk.LEFT, padx=5)

        ttk.Label(control_frame, text="Powiększenie:").pack(side=tk.LEFT, padx=(20, 5))
        ttk.Button(control_frame, text="-", command=self.zoomOut, width=3).pack(side=tk.LEFT)
        self.zoom_label = ttk.Label(control_frame, text="100%")
        self.zoom_label.pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="+", command=self.zoomIn, width=3).pack(side=tk.LEFT)
        ttk.Button(control_frame, text="Reset", command=self.resetView).pack(side=tk.LEFT, padx=5)

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
                self.original_image = np.array(img.convert('RGB'))
                self.metadata = {
                    'format': 'JPEG',
                    'width': img.width,
                    'height': img.height
                }
                self.current_filepath = filepath

            else:
                messagebox.showerror("Błąd", f"Nieobsługiwany format pliku: .{ext}")
                return

            self.processed_image = self.original_image.copy()
            self.zoom_level = 1.0
            self.pan_x = 0
            self.pan_y = 0
            self.color_scale = 1.0
            self.color_scale_var.set(1.0)

            self.updateDisplay()
            self.updateInfo()

        except Exception as e:
            messagebox.showerror("Błąd", f"Nie udało się wczytać obrazu:\n{str(e)}")

    def resetImage(self):
        if self.original_image is None:
            messagebox.showwarning("Uwaga", "Najpierw wczytaj obraz!")
            return

        self.processed_image = self.original_image.copy()
        self.updateDisplay()
        messagebox.showinfo("Info", "Obraz zresetowany do oryginału")

    def saveAsJpeg(self):
        if self.processed_image is None:
            messagebox.showwarning("Uwaga", "Najpierw wczytaj obraz!")
            return

        filepath = filedialog.asksaveasfilename(
            title="Zapisz jako JPEG",
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("JPEG", "*.jpeg")]
        )

        if not filepath:
            return

        quality_dialog = tk.Toplevel(self.root)
        quality_dialog.title("Jakość kompresji JPEG")
        quality_dialog.geometry("300x150")
        quality_dialog.transient(self.root)
        quality_dialog.grab_set()

        ttk.Label(quality_dialog, text="Wybierz jakość kompresji (1-100):").pack(pady=10)

        quality_var = tk.IntVar(value=50)
        quality_scale = ttk.Scale(quality_dialog, from_=1, to=100, orient=tk.HORIZONTAL,
                                  variable=quality_var, length=250)
        quality_scale.pack(pady=10)

        quality_label = ttk.Label(quality_dialog, text="50")
        quality_label.pack()

        def updateLabel(val):
            quality_label.config(text=f"{int(float(val))}")

        quality_scale.config(command=updateLabel)

        result = {'save': False, 'quality': 50}

        def onSave():
            result['save'] = True
            result['quality'] = quality_var.get()
            quality_dialog.destroy()

        def onCancel():
            quality_dialog.destroy()

        button_frame = ttk.Frame(quality_dialog)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="Zapisz", command=onSave).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Anuluj", command=onCancel).pack(side=tk.LEFT, padx=5)

        self.root.wait_window(quality_dialog)

        if result['save']:
            try:
                img_to_save = self.applyColorScaling(self.processed_image)
                img = Image.fromarray(img_to_save)
                img.save(filepath, 'JPEG', quality=result['quality'])
                messagebox.showinfo("Sukces", f"Obraz zapisany jako:\n{filepath}\nJakość: {result['quality']}")
            except Exception as e:
                messagebox.showerror("Błąd", f"Nie udało się zapisać obrazu:\n{str(e)}")

    def addValueDialog(self):
        if self.processed_image is None:
            messagebox.showwarning("Uwaga", "Najpierw wczytaj obraz!")
            return

        value = self.getValueFromUser("Dodawanie", "Podaj wartość do dodania (-255 do 255):", -255, 255)
        if value is not None:
            self.processed_image = ImageProcessor.add_value(self.processed_image, value)
            self.updateDisplay()

    def subtractValueDialog(self):
        if self.processed_image is None:
            messagebox.showwarning("Uwaga", "Najpierw wczytaj obraz!")
            return

        value = self.getValueFromUser("Odejmowanie", "Podaj wartość do odjęcia (-255 do 255):", -255, 255)
        if value is not None:
            self.processed_image = ImageProcessor.subtract_value(self.processed_image, value)
            self.updateDisplay()

    def multiplyValueDialog(self):
        if self.processed_image is None:
            messagebox.showwarning("Uwaga", "Najpierw wczytaj obraz!")
            return

        value = self.getValueFromUser("Mnożenie", "Podaj wartość mnożnika (0.1 do 10):", 0.1, 10.0)
        if value is not None:
            self.processed_image = ImageProcessor.multiply_value(self.processed_image, value)
            self.updateDisplay()

    def divideValueDialog(self):
        if self.processed_image is None:
            messagebox.showwarning("Uwaga", "Najpierw wczytaj obraz!")
            return

        value = self.getValueFromUser("Dzielenie", "Podaj dzielnik (0.1 do 10):", 0.1, 10.0)
        if value is not None and value != 0:
            self.processed_image = ImageProcessor.divide_value(self.processed_image, value)
            self.updateDisplay()

    def brightnessDialog(self):
        if self.processed_image is None:
            messagebox.showwarning("Uwaga", "Najpierw wczytaj obraz!")
            return

        value = self.getValueFromUser("Zmiana jasności", "Podaj poziom jasności (-100 do 100):", -100, 100)
        if value is not None:
            self.processed_image = ImageProcessor.change_brightness(self.processed_image, value)
            self.updateDisplay()

    def toGrayscaleAverage(self):
        if self.processed_image is None:
            messagebox.showwarning("Uwaga", "Najpierw wczytaj obraz!")
            return

        self.processed_image = ImageProcessor.to_grayscale_average(self.processed_image)
        self.updateDisplay()

    def toGrayscaleLuminosity(self):
        if self.processed_image is None:
            messagebox.showwarning("Uwaga", "Najpierw wczytaj obraz!")
            return

        self.processed_image = ImageProcessor.to_grayscale_luminosity(self.processed_image)
        self.updateDisplay()

    def applySmoothingFilter(self):
        if self.processed_image is None:
            messagebox.showwarning("Uwaga", "Najpierw wczytaj obraz!")
            return

        messagebox.showinfo("Przetwarzanie", "Przetwarzanie obrazu...\nTo może chwilę potrwać.")
        self.root.update()
        self.processed_image = ImageProcessor.smoothing_filter(self.processed_image, size=3)
        self.updateDisplay()

    def applyMedianFilter(self):
        if self.processed_image is None:
            messagebox.showwarning("Uwaga", "Najpierw wczytaj obraz!")
            return

        messagebox.showinfo("Przetwarzanie", "Przetwarzanie obrazu...\nTo może chwilę potrwać.")
        self.root.update()
        self.processed_image = ImageProcessor.median_filter(self.processed_image, size=3)
        self.updateDisplay()

    def applySobelFilter(self):
        if self.processed_image is None:
            messagebox.showwarning("Uwaga", "Najpierw wczytaj obraz!")
            return

        messagebox.showinfo("Przetwarzanie", "Przetwarzanie obrazu...\nTo może chwilę potrwać.")
        self.root.update()
        self.processed_image = ImageProcessor.sobel_filter(self.processed_image)
        self.updateDisplay()

    def applySharpeningFilter(self):
        if self.processed_image is None:
            messagebox.showwarning("Uwaga", "Najpierw wczytaj obraz!")
            return

        messagebox.showinfo("Przetwarzanie", "Przetwarzanie obrazu...")
        self.root.update()
        self.processed_image = ImageProcessor.sharpening_filter(self.processed_image)
        self.updateDisplay()

    def applyGaussianBlur(self):
        if self.processed_image is None:
            messagebox.showwarning("Uwaga", "Najpierw wczytaj obraz!")
            return

        messagebox.showinfo("Przetwarzanie", "Przetwarzanie obrazu...\nTo może chwilę potrwać.")
        self.root.update()
        self.processed_image = ImageProcessor.gaussian_blur(self.processed_image, size=3, sigma=1.4)
        self.updateDisplay()

    def getValueFromUser(self, title: str, prompt: str, min_val: float, max_val: float, default: float = 0) -> Optional[
        float]:
        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.geometry("400x150")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text=prompt).pack(pady=10)

        value_var = tk.DoubleVar(value=default)
        entry = ttk.Entry(dialog, textvariable=value_var, width=20)
        entry.pack(pady=5)
        entry.focus()

        result = {'value': None}

        def on_ok():
            try:
                val = value_var.get()
                if min_val <= val <= max_val:
                    result['value'] = val
                    dialog.destroy()
                else:
                    messagebox.showerror("Błąd", f"Wartość musi być między {min_val} a {max_val}")
            except ValueError:
                messagebox.showerror("Błąd", "Nieprawidłowa wartość")

        def on_cancel():
            dialog.destroy()

        def on_enter(event):
            on_ok()

        entry.bind("<Return>", on_enter)

        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="OK", command=on_ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Anuluj", command=on_cancel).pack(side=tk.LEFT, padx=5)

        self.root.wait_window(dialog)
        return result['value']

    def applyColorScaling(self, image: np.ndarray) -> np.ndarray:
        if self.color_scale == 1.0:
            return image

        scaled = image.astype(np.float32) * self.color_scale
        return np.clip(scaled, 0, 255).astype(np.uint8)

    def onColorScaleChange(self, val):
        self.color_scale = float(val)
        self.color_scale_label.config(text=f"{self.color_scale:.2f}x")
        self.updateDisplay()

    def zoomIn(self):
        self.zoom_level = min(self.zoom_level * 1.5, 20.0)
        self.updateDisplay()

    def zoomOut(self):
        self.zoom_level = max(self.zoom_level / 1.5, 0.1)
        self.updateDisplay()

    def resetView(self):
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.updateDisplay()

    def updateDisplay(self):
        if self.processed_image is None:
            return

        scaled_image = self.applyColorScaling(self.processed_image)

        h, w = scaled_image.shape[:2]
        new_w = int(w * self.zoom_level)
        new_h = int(h * self.zoom_level)

        img = Image.fromarray(scaled_image)

        if self.zoom_level >= 1.0:
            img = img.resize((new_w, new_h), Image.NEAREST)
        else:
            img = img.resize((new_w, new_h), Image.LANCZOS)

        self.display_image = np.array(img)
        self.photo_image = ImageTk.PhotoImage(img)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_image)

        self.canvas.config(scrollregion=(0, 0, new_w, new_h))

        self.zoom_label.config(text=f"{int(self.zoom_level * 100)}%")

    def onMousePress(self, event):
        self.is_panning = True
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        self.canvas.config(cursor="fleur")

    def onMouseDrag(self, event):
        if self.is_panning and self.zoom_level > 1.0:
            dx = event.x - self.pan_start_x
            dy = event.y - self.pan_start_y

            self.canvas.xview_scroll(int(-dx), "units")
            self.canvas.yview_scroll(int(-dy), "units")

            self.pan_start_x = event.x
            self.pan_start_y = event.y

    def onMouseRelease(self, event):
        self.is_panning = False
        self.canvas.config(cursor="")

    def onMouseMove(self, event):
        if self.original_image is None or self.zoom_level < 3.0:
            return

        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        orig_x = int(canvas_x / self.zoom_level)
        orig_y = int(canvas_y / self.zoom_level)

        h, w = self.original_image.shape[:2]

        if 0 <= orig_x < w and 0 <= orig_y < h:
            r, g, b = self.original_image[orig_y, orig_x]
            self.root.title(f"Przeglądarka obrazów - Pozycja ({orig_x}, {orig_y}) - RGB({r}, {g}, {b})")
        else:
            self.root.title("Przeglądarka obrazów PPM/JPEG")

    def onMousewheel(self, event):
        if event.delta > 0:
            self.zoomIn()
        else:
            self.zoomOut()

    def updateInfo(self):
        if self.original_image is None:
            self.info_label.config(text="Nie załadowano obrazu")
            return

        info = f"Plik: {self.current_filepath} | "
        info += f"Format: {self.metadata.get('format', 'N/A')} | "
        info += f"Rozmiar: {self.metadata.get('width', 0)}x{self.metadata.get('height', 0)}"

        if 'maxval' in self.metadata:
            info += f" | Max wartość: {self.metadata['maxval']}"

        self.info_label.config(text=info)


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageViewer(root)
    root.mainloop()