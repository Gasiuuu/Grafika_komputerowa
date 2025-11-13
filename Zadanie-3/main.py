import tkinter as tk
from tkinter import ttk
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class ColorConverterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Konwerter RGB - CMYK / Kostka 3D")
        self.root.geometry("1050x800")

        self.r_var = tk.IntVar(value=128)
        self.g_var = tk.IntVar(value=128)
        self.b_var = tk.IntVar(value=128)

        self.c_var = tk.DoubleVar(value=0.0)
        self.m_var = tk.DoubleVar(value=0.0)
        self.y_var = tk.DoubleVar(value=0.0)
        self.k_var = tk.DoubleVar(value=50.0)

        self.updating = False

        self.elevation = 30
        self.azimuth = 45
        self.slice_plane = 'none'
        self.slice_value = 128

        self.create_widgets()
        self.rgb_to_cmyk()
        self.update_cube()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        rgb_frame = ttk.LabelFrame(main_frame, text="RGB", padding="10")
        rgb_frame.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.N, tk.W, tk.E))

        self.create_rgb_controls(rgb_frame)

        preview_frame = ttk.LabelFrame(main_frame, text="Podgląd koloru", padding="10")
        preview_frame.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.N, tk.W, tk.E))

        self.color_preview = tk.Canvas(preview_frame, width=150, height=150, bg='gray')
        self.color_preview.pack()

        cmyk_frame = ttk.LabelFrame(main_frame, text="CMYK", padding="10")
        cmyk_frame.grid(row=0, column=2, padx=5, pady=5, sticky=(tk.N, tk.W, tk.E))

        self.create_cmyk_controls(cmyk_frame)

        cube_frame = ttk.LabelFrame(main_frame, text="Kostka RGB", padding="10")
        cube_frame.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.create_3d_cube(cube_frame)

    def create_rgb_controls(self, parent):
        red_frame = ttk.Frame(parent)
        red_frame.grid(row=0, column=0, sticky=tk.W, pady=5)
        red_square = tk.Canvas(red_frame, width=15, height=15, bg='red', highlightthickness=1,
                               highlightbackground='black')
        red_square.pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(red_frame, text="Red (R):").pack(side=tk.LEFT)
        r_scale = ttk.Scale(parent, from_=0, to=255, variable=self.r_var,
                            orient=tk.HORIZONTAL, length=200, command=self.on_rgb_change)
        r_scale.grid(row=0, column=1, padx=5, pady=5)
        r_entry = ttk.Entry(parent, textvariable=self.r_var, width=10)
        r_entry.grid(row=0, column=2, padx=5, pady=5)
        r_entry.bind('<Return>', self.on_rgb_change)

        green_frame = ttk.Frame(parent)
        green_frame.grid(row=1, column=0, sticky=tk.W, pady=5)
        green_square = tk.Canvas(green_frame, width=15, height=15, bg='green', highlightthickness=1,
                                 highlightbackground='black')
        green_square.pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(green_frame, text="Green (G):").pack(side=tk.LEFT)
        g_scale = ttk.Scale(parent, from_=0, to=255, variable=self.g_var,
                            orient=tk.HORIZONTAL, length=200, command=self.on_rgb_change)
        g_scale.grid(row=1, column=1, padx=5, pady=5)
        g_entry = ttk.Entry(parent, textvariable=self.g_var, width=10)
        g_entry.grid(row=1, column=2, padx=5, pady=5)
        g_entry.bind('<Return>', self.on_rgb_change)

        blue_frame = ttk.Frame(parent)
        blue_frame.grid(row=2, column=0, sticky=tk.W, pady=5)
        blue_square = tk.Canvas(blue_frame, width=15, height=15, bg='blue', highlightthickness=1,
                                highlightbackground='black')
        blue_square.pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(blue_frame, text="Blue (B):").pack(side=tk.LEFT)
        b_scale = ttk.Scale(parent, from_=0, to=255, variable=self.b_var,
                            orient=tk.HORIZONTAL, length=200, command=self.on_rgb_change)
        b_scale.grid(row=2, column=1, padx=5, pady=5)
        b_entry = ttk.Entry(parent, textvariable=self.b_var, width=10)
        b_entry.grid(row=2, column=2, padx=5, pady=5)
        b_entry.bind('<Return>', self.on_rgb_change)

    def create_cmyk_controls(self, parent):
        cyan_frame = ttk.Frame(parent)
        cyan_frame.grid(row=0, column=0, sticky=tk.W, pady=5)
        cyan_square = tk.Canvas(cyan_frame, width=15, height=15, bg='cyan', highlightthickness=1,
                                highlightbackground='black')
        cyan_square.pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(cyan_frame, text="Cyan (C %):").pack(side=tk.LEFT)
        c_scale = ttk.Scale(parent, from_=0, to=100, variable=self.c_var,
                            orient=tk.HORIZONTAL, length=200, command=self.on_cmyk_change)
        c_scale.grid(row=0, column=1, padx=5, pady=5)
        c_entry = ttk.Entry(parent, textvariable=self.c_var, width=10)
        c_entry.grid(row=0, column=2, padx=5, pady=5)
        c_entry.bind('<Return>', self.on_cmyk_change)

        magenta_frame = ttk.Frame(parent)
        magenta_frame.grid(row=1, column=0, sticky=tk.W, pady=5)
        magenta_square = tk.Canvas(magenta_frame, width=15, height=15, bg='magenta', highlightthickness=1,
                                   highlightbackground='black')
        magenta_square.pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(magenta_frame, text="Magenta (M %):").pack(side=tk.LEFT)
        m_scale = ttk.Scale(parent, from_=0, to=100, variable=self.m_var,
                            orient=tk.HORIZONTAL, length=200, command=self.on_cmyk_change)
        m_scale.grid(row=1, column=1, padx=5, pady=5)
        m_entry = ttk.Entry(parent, textvariable=self.m_var, width=10)
        m_entry.grid(row=1, column=2, padx=5, pady=5)
        m_entry.bind('<Return>', self.on_cmyk_change)

        yellow_frame = ttk.Frame(parent)
        yellow_frame.grid(row=2, column=0, sticky=tk.W, pady=5)
        yellow_square = tk.Canvas(yellow_frame, width=15, height=15, bg='yellow', highlightthickness=1,
                                  highlightbackground='black')
        yellow_square.pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(yellow_frame, text="Yellow (Y %):").pack(side=tk.LEFT)
        y_scale = ttk.Scale(parent, from_=0, to=100, variable=self.y_var,
                            orient=tk.HORIZONTAL, length=200, command=self.on_cmyk_change)
        y_scale.grid(row=2, column=1, padx=5, pady=5)
        y_entry = ttk.Entry(parent, textvariable=self.y_var, width=10)
        y_entry.grid(row=2, column=2, padx=5, pady=5)
        y_entry.bind('<Return>', self.on_cmyk_change)

        key_frame = ttk.Frame(parent)
        key_frame.grid(row=3, column=0, sticky=tk.W, pady=5)
        key_square = tk.Canvas(key_frame, width=15, height=15, bg='black', highlightthickness=1,
                               highlightbackground='black')
        key_square.pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(key_frame, text="Key/Black (K %):").pack(side=tk.LEFT)
        k_scale = ttk.Scale(parent, from_=0, to=100, variable=self.k_var,
                            orient=tk.HORIZONTAL, length=200, command=self.on_cmyk_change)
        k_scale.grid(row=3, column=1, padx=5, pady=5)
        k_entry = ttk.Entry(parent, textvariable=self.k_var, width=10)
        k_entry.grid(row=3, column=2, padx=5, pady=5)
        k_entry.bind('<Return>', self.on_cmyk_change)

    def create_3d_cube(self, parent):
        slice_frame = ttk.Frame(parent)
        slice_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        ttk.Label(slice_frame, text="Przekrój:").pack(side=tk.LEFT, padx=5)

        self.slice_var = tk.StringVar(value='none')
        slice_options = [('Brak', 'none'), ('X (Red)', 'x'), ('Y (Green)', 'y'), ('Z (Blue)', 'z')]

        for text, value in slice_options:
            ttk.Radiobutton(slice_frame, text=text, variable=self.slice_var,
                            value=value, command=self.update_cube).pack(side=tk.LEFT, padx=2)

        ttk.Label(slice_frame, text="Pozycja:").pack(side=tk.LEFT, padx=5)
        self.slice_scale = ttk.Scale(slice_frame, from_=0, to=255, orient=tk.HORIZONTAL,
                                     length=150, command=self.on_slice_change)
        self.slice_scale.set(128)
        self.slice_scale.pack(side=tk.LEFT, padx=5)

        fig_frame = ttk.Frame(parent)
        fig_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(6, 5))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(self.fig, master=fig_frame)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig_slice = Figure(figsize=(4, 4))
        self.ax_slice = self.fig_slice.add_subplot(111)
        self.canvas_slice = FigureCanvasTkAgg(self.fig_slice, master=fig_frame)
        self.canvas_slice.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.draw_rgb_cube()

    def rgb_to_cmyk(self, event=None):
        if self.updating:
            return
        self.updating = True

        try:
            r = max(0, min(255, int(self.r_var.get())))
            g = max(0, min(255, int(self.g_var.get())))
            b = max(0, min(255, int(self.b_var.get())))

            r_norm = r / 255.0
            g_norm = g / 255.0
            b_norm = b / 255.0

            k = min(1 - r_norm, 1 - g_norm, 1 - b_norm)

            if k == 1:
                c = m = y = 0
            else:
                c = (1 - r_norm - k) / (1 - k)
                m = (1 - g_norm - k) / (1 - k)
                y = (1 - b_norm - k) / (1 - k)

            self.c_var.set(round(c * 100))
            self.m_var.set(round(m * 100))
            self.y_var.set(round(y * 100))
            self.k_var.set(round(k * 100))

            self.update_color_preview()
        finally:
            self.updating = False

    def cmyk_to_rgb(self, event=None):
        if self.updating:
            return
        self.updating = True

        try:
            c = max(0, min(100, float(self.c_var.get()))) / 100.0
            m = max(0, min(100, float(self.m_var.get()))) / 100.0
            y = max(0, min(100, float(self.y_var.get()))) / 100.0
            k = max(0, min(100, float(self.k_var.get()))) / 100.0

            r = 1 - min(1, c * (1 - k) + k)
            g = 1 - min(1, m * (1 - k) + k)
            b = 1 - min(1, y * (1 - k) + k)

            self.r_var.set(int(round(r * 255)))
            self.g_var.set(int(round(g * 255)))
            self.b_var.set(int(round(b * 255)))

            self.update_color_preview()
        finally:
            self.updating = False

    def on_rgb_change(self, event=None):
        if not self.updating:
            self.r_var.set(int(round(self.r_var.get())))
            self.g_var.set(int(round(self.g_var.get())))
            self.b_var.set(int(round(self.b_var.get())))
            self.rgb_to_cmyk()

    def on_cmyk_change(self, event=None):
        if not self.updating:
            self.c_var.set(int(round(self.c_var.get())))
            self.m_var.set(int(round(self.m_var.get())))
            self.y_var.set(int(round(self.y_var.get())))
            self.k_var.set(int(round(self.k_var.get())))
            self.cmyk_to_rgb()

    def update_color_preview(self):
        r = int(self.r_var.get())
        g = int(self.g_var.get())
        b = int(self.b_var.get())

        color = f'#{r:02x}{g:02x}{b:02x}'
        self.color_preview.configure(bg=color)

    def draw_rgb_cube(self):
        self.ax.clear()

        resolution = 12

        for i in range(resolution):
            for j in range(resolution):
                t1 = i / resolution
                t2 = (i + 1) / resolution
                s1 = j / resolution
                s2 = (j + 1) / resolution

                vertices = [
                    [t1 * 255, s1 * 255, 255],
                    [t2 * 255, s1 * 255, 255],
                    [t2 * 255, s2 * 255, 255],
                    [t1 * 255, s2 * 255, 255]
                ]
                self.add_quad(vertices)

                vertices = [
                    [t1 * 255, s1 * 255, 0],
                    [t2 * 255, s1 * 255, 0],
                    [t2 * 255, s2 * 255, 0],
                    [t1 * 255, s2 * 255, 0]
                ]
                self.add_quad(vertices)

                vertices = [
                    [255, t1 * 255, s1 * 255],
                    [255, t2 * 255, s1 * 255],
                    [255, t2 * 255, s2 * 255],
                    [255, t1 * 255, s2 * 255]
                ]
                self.add_quad(vertices)

                vertices = [
                    [0, t1 * 255, s1 * 255],
                    [0, t2 * 255, s1 * 255],
                    [0, t2 * 255, s2 * 255],
                    [0, t1 * 255, s2 * 255]
                ]
                self.add_quad(vertices)

                vertices = [
                    [t1 * 255, 255, s1 * 255],
                    [t2 * 255, 255, s1 * 255],
                    [t2 * 255, 255, s2 * 255],
                    [t1 * 255, 255, s2 * 255]
                ]
                self.add_quad(vertices)

                vertices = [
                    [t1 * 255, 0, s1 * 255],
                    [t2 * 255, 0, s1 * 255],
                    [t2 * 255, 0, s2 * 255],
                    [t1 * 255, 0, s2 * 255]
                ]
                self.add_quad(vertices)

        self.ax.set_xlim(0, 255)
        self.ax.set_ylim(0, 255)
        self.ax.set_zlim(0, 255)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_zticks([])
        self.ax.set_xlabel('')
        self.ax.set_ylabel('')
        self.ax.set_zlabel('')
        self.ax.view_init(elev=self.elevation, azim=self.azimuth)

        self.canvas.draw()

    def add_quad(self, vertices):
        avg_color = np.mean(vertices, axis=0) / 255.0

        poly = Poly3DCollection([vertices], alpha=0.8)
        poly.set_facecolor(avg_color)
        poly.set_edgecolor('none')
        self.ax.add_collection3d(poly)

    def draw_slice(self):
        plane = self.slice_var.get()
        value = self.slice_scale.get()

        resolution = 256

        if plane == 'x':
            y = np.linspace(0, 255, resolution)
            z = np.linspace(0, 255, resolution)
            Y, Z = np.meshgrid(y, z)
            colors = np.zeros((*Y.shape, 3))
            colors[:, :, 0] = value / 255.0
            colors[:, :, 1] = Y / 255.0
            colors[:, :, 2] = Z / 255.0

        elif plane == 'y':
            x = np.linspace(0, 255, resolution)
            z = np.linspace(0, 255, resolution)
            X, Z = np.meshgrid(x, z)
            colors = np.zeros((*X.shape, 3))
            colors[:, :, 0] = X / 255.0
            colors[:, :, 1] = value / 255.0
            colors[:, :, 2] = Z / 255.0

        else:
            x = np.linspace(0, 255, resolution)
            y = np.linspace(0, 255, resolution)
            X, Y = np.meshgrid(x, y)
            colors = np.zeros((*X.shape, 3))
            colors[:, :, 0] = X / 255.0
            colors[:, :, 1] = Y / 255.0
            colors[:, :, 2] = value / 255.0

        self.draw_2d_slice(plane, value, colors)

    def draw_2d_slice(self, plane, value, colors):
        self.ax_slice.clear()

        if plane == 'x':
            self.ax_slice.imshow(colors, origin='lower', extent=[0, 255, 0, 255])
            self.ax_slice.set_title(f'Przekrój X (Czerwony = {int(value)})')
        elif plane == 'y':
            self.ax_slice.imshow(colors, origin='lower', extent=[0, 255, 0, 255])
            self.ax_slice.set_title(f'Przekrój Y (Zielony = {int(value)})')
        else:
            self.ax_slice.imshow(colors, origin='lower', extent=[0, 255, 0, 255])
            self.ax_slice.set_title(f'Przekrój Z (Niebieski = {int(value)})')

        self.ax_slice.set_xticks([])
        self.ax_slice.set_yticks([])
        self.ax_slice.set_xlabel('')
        self.ax_slice.set_ylabel('')

        self.canvas_slice.draw()

    def on_slice_change(self, event=None):
        if self.slice_var.get() != 'none':
            self.draw_slice()

    def update_cube(self):
        if self.slice_var.get() != 'none':
            self.draw_slice()
        else:
            self.ax_slice.clear()
            self.ax_slice.set_xlim(0, 255)
            self.ax_slice.set_ylim(0, 255)
            self.ax_slice.set_aspect('equal')
            self.ax_slice.set_xticks([])
            self.ax_slice.set_yticks([])
            self.canvas_slice.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = ColorConverterApp(root)
    root.mainloop()