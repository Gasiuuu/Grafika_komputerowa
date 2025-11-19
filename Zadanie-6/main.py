import tkinter as tk
from tkinter import ttk
import math


class BezierCurveEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Edytor Krzywych Beziera")

        self.degree = 3
        self.points = []
        self.dragging_point = None
        self.point_radius = 6
        self.curve_resolution = 100

        self.setup_ui()
        self.initialize_default_points()
        self.draw()

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        control_frame = ttk.LabelFrame(main_frame, text="Ustawienia", padding="10")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Label(control_frame, text="Stopień krzywej:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.degree_var = tk.StringVar(value=str(self.degree))
        degree_spinbox = ttk.Spinbox(control_frame, from_=1, to=20, textvariable=self.degree_var, width=10)
        degree_spinbox.grid(row=0, column=1, sticky=tk.W, padx=(0, 10))

        ttk.Button(control_frame, text="Zastosuj", command=self.apply_degree).grid(row=0, column=2, padx=(0, 10))
        ttk.Button(control_frame, text="Wyczyść", command=self.clear_points).grid(row=0, column=3, padx=(0, 10))
        ttk.Button(control_frame, text="Resetuj", command=self.reset_points).grid(row=0, column=4)

        self.canvas = tk.Canvas(main_frame, width=800, height=500, bg="white", cursor="crosshair")
        self.canvas.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        points_frame = ttk.LabelFrame(main_frame, text="Punkty kontrolne", padding="10")
        points_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0))

        points_canvas = tk.Canvas(points_frame, width=250, height=500)
        scrollbar = ttk.Scrollbar(points_frame, orient="vertical", command=points_canvas.yview)
        self.points_inner_frame = ttk.Frame(points_canvas)

        self.points_inner_frame.bind(
            "<Configure>",
            lambda e: points_canvas.configure(scrollregion=points_canvas.bbox("all"))
        )

        points_canvas.create_window((0, 0), window=self.points_inner_frame, anchor="nw")
        points_canvas.configure(yscrollcommand=scrollbar.set)

        points_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        info_frame = ttk.Frame(main_frame)
        info_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))

        self.info_label = ttk.Label(info_frame, text="Kliknij na canvas, aby dodać punkty kontrolne")
        self.info_label.pack()

        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=3)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)

    def initialize_default_points(self):
        self.points = [
            [100, 400],
            [200, 100],
            [400, 100],
            [500, 400]
        ]
        self.update_points_panel()

    def apply_degree(self):
        try:
            new_degree = int(self.degree_var.get())
            if new_degree < 1:
                raise ValueError
            self.degree = new_degree
            self.clear_points()
            self.update_info(f"Ustaw {self.degree + 1} punktów kontrolnych dla krzywej stopnia {self.degree}")
        except ValueError:
            self.degree_var.set(str(self.degree))
            self.update_info("Błędny stopień krzywej!")

    def clear_points(self):
        self.points = []
        self.update_points_panel()
        self.draw()
        self.update_info("Kliknij na canvas, aby dodać punkty kontrolne")

    def reset_points(self):
        self.degree = 3
        self.degree_var.set("3")
        self.initialize_default_points()
        self.draw()
        self.update_info("Zresetowano do domyślnej krzywej")

    def on_canvas_click(self, event):
        for i, point in enumerate(self.points):
            if self.is_near_point(event.x, event.y, point):
                self.dragging_point = i
                return

        if len(self.points) < self.degree + 1:
            self.points.append([event.x, event.y])
            self.update_points_panel()
            self.draw()

            remaining = self.degree + 1 - len(self.points)
            if remaining > 0:
                self.update_info(f"Dodano punkt. Pozostało {remaining} punktów do dodania")
            else:
                self.update_info("Wszystkie punkty dodane. Możesz je teraz edytować")

    def on_canvas_drag(self, event):
        if self.dragging_point is not None:
            x = max(0, min(event.x, self.canvas.winfo_width()))
            y = max(0, min(event.y, self.canvas.winfo_height()))

            self.points[self.dragging_point] = [x, y]
            self.update_point_entry(self.dragging_point)
            self.draw()

    def on_canvas_release(self, event):
        self.dragging_point = None

    def is_near_point(self, x, y, point):
        dx = x - point[0]
        dy = y - point[1]
        distance = math.sqrt(dx * dx + dy * dy)
        return distance <= self.point_radius + 5

    def update_points_panel(self):
        for widget in self.points_inner_frame.winfo_children():
            widget.destroy()

        for i, point in enumerate(self.points):
            frame = ttk.Frame(self.points_inner_frame)
            frame.pack(fill=tk.X, pady=5)

            ttk.Label(frame, text=f"P{i}:", width=4).pack(side=tk.LEFT)

            ttk.Label(frame, text="X:").pack(side=tk.LEFT, padx=(5, 2))
            x_var = tk.StringVar(value=str(int(point[0])))
            x_entry = ttk.Entry(frame, textvariable=x_var, width=8)
            x_entry.pack(side=tk.LEFT, padx=(0, 5))
            x_entry.bind("<Return>", lambda e, idx=i: self.update_point_from_entry(idx))
            x_entry.bind("<FocusOut>", lambda e, idx=i: self.update_point_from_entry(idx))

            ttk.Label(frame, text="Y:").pack(side=tk.LEFT, padx=(5, 2))
            y_var = tk.StringVar(value=str(int(point[1])))
            y_entry = ttk.Entry(frame, textvariable=y_var, width=8)
            y_entry.pack(side=tk.LEFT, padx=(0, 5))
            y_entry.bind("<Return>", lambda e, idx=i: self.update_point_from_entry(idx))
            y_entry.bind("<FocusOut>", lambda e, idx=i: self.update_point_from_entry(idx))

            ttk.Button(frame, text="×", width=3, command=lambda idx=i: self.remove_point(idx)).pack(side=tk.LEFT)

            setattr(self, f"point_{i}_x_var", x_var)
            setattr(self, f"point_{i}_y_var", y_var)

    def update_point_entry(self, index):
        if hasattr(self, f"point_{index}_x_var"):
            x_var = getattr(self, f"point_{index}_x_var")
            y_var = getattr(self, f"point_{index}_y_var")
            x_var.set(str(int(self.points[index][0])))
            y_var.set(str(int(self.points[index][1])))

    def update_point_from_entry(self, index):
        try:
            x_var = getattr(self, f"point_{index}_x_var")
            y_var = getattr(self, f"point_{index}_y_var")

            x = float(x_var.get())
            y = float(y_var.get())

            self.points[index] = [x, y]
            self.draw()
        except (ValueError, AttributeError):
            self.update_point_entry(index)

    def remove_point(self, index):
        if 0 <= index < len(self.points):
            self.points.pop(index)
            self.update_points_panel()
            self.draw()
            self.update_info(f"Usunięto punkt. Pozostało {len(self.points)} punktów")

    def de_casteljau(self, t):
        if not self.points:
            return 0, 0

        temp_points = [point[:] for point in self.points]
        n = len(temp_points)

        for r in range(1, n):
            for i in range(n - r):
                temp_points[i][0] = (1 - t) * temp_points[i][0] + t * temp_points[i + 1][0]
                temp_points[i][1] = (1 - t) * temp_points[i][1] + t * temp_points[i + 1][1]

        return temp_points[0][0], temp_points[0][1]

    def draw(self):
        self.canvas.delete("all")

        if len(self.points) < 2:
            return

        for i in range(len(self.points) - 1):
            self.canvas.create_line(
                self.points[i][0], self.points[i][1],
                self.points[i + 1][0], self.points[i + 1][1],
                fill="lightgray", width=1, dash=(4, 4)
            )

        if len(self.points) == self.degree + 1:
            curve_points = []
            for i in range(self.curve_resolution + 1):
                t = i / self.curve_resolution
                x, y = self.de_casteljau(t)
                curve_points.extend([x, y])

            if len(curve_points) >= 4:
                self.canvas.create_line(curve_points, fill="blue", width=2, smooth=True)

        for i, point in enumerate(self.points):
            self.canvas.create_oval(
                point[0] - self.point_radius,
                point[1] - self.point_radius,
                point[0] + self.point_radius,
                point[1] + self.point_radius,
                fill="red", width=2
            )

            self.canvas.create_text(
                point[0], point[1] - self.point_radius - 10,
                text=f"P{i}", fill="black", font=("Arial", 10, "bold")
            )

    def update_info(self, text):
        self.info_label.config(text=text)


if __name__ == "__main__":
    root = tk.Tk()
    app = BezierCurveEditor(root)
    root.mainloop()