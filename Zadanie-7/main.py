import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import math
from typing import List, Tuple


class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def distance_to(self, other: 'Point') -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


class Transformations:
    @staticmethod
    def translate(point: Point, h: float, v: float) -> Point:
        return Point(point.x + h, point.y + v)

    @staticmethod
    def rotate(point: Point, pivot: Point, angle_rad: float) -> Point:
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        dx = point.x - pivot.x
        dy = point.y - pivot.y

        x_new = pivot.x + dx * cos_a - dy * sin_a
        y_new = pivot.y + dx * sin_a + dy * cos_a

        return Point(x_new, y_new)

    @staticmethod
    def scale(point: Point, pivot: Point, k: float) -> Point:
        x_new = point.x * k + (1 - k) * pivot.x
        y_new = point.y * k + (1 - k) * pivot.y

        return Point(x_new, y_new)


class Polygon:
    def __init__(self, points: List[Point], color: str = ""):
        self.points = points
        self.color = color
        self.canvas_id = None

    def translate(self, h: float, v: float):
        self.points = [Transformations.translate(p, h, v) for p in self.points]

    def rotate(self, pivot: Point, angle_rad: float):
        self.points = [Transformations.rotate(p, pivot, angle_rad) for p in self.points]

    def scale(self, pivot: Point, k: float):
        self.points = [Transformations.scale(p, pivot, k) for p in self.points]

    def contains_point(self, point: Point, canvas) -> bool:
        if self.canvas_id is None:
            return False
        items = canvas.find_overlapping(point.x - 1, point.y - 1, point.x + 1, point.y + 1)

        return self.canvas_id in items

    def to_dict(self) -> dict:
        return {
            'points': [[p.x, p.y] for p in self.points],
            'color': self.color
        }

    @staticmethod
    def from_dict(data: dict) -> 'Polygon':
        points = [Point(p[0], p[1]) for p in data['points']]
        return Polygon(points, data.get('color', 'blue'))


class PolygonEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Edytor Figur 2D - Transformacje Jednorodne")

        self.polygons: List[Polygon] = []
        self.current_points: List[Point] = []
        self.selected_polygon: Polygon = None
        self.mode = "draw"
        self.pivot_point: Point = None
        self.drag_start: Point = None

        self.setup_ui()

    def setup_ui(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Plik", menu=file_menu)
        file_menu.add_command(label="Zapisz", command=self.save_to_file)
        file_menu.add_command(label="Wczytaj", command=self.load_from_file)
        file_menu.add_separator()
        file_menu.add_command(label="Wyjście", command=self.root.quit)

        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.canvas = tk.Canvas(main_frame, width=800, height=600, bg="white", cursor="cross")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        control_frame = ttk.Frame(main_frame, width=300)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5)

        ttk.Label(control_frame, text="Tryb:", font=("Arial", 12, "bold")).pack(pady=5)

        self.mode_var = tk.StringVar(value="draw")
        modes = [
            ("Rysowanie", "draw"),
            ("Przesunięcie", "translate"),
            ("Obrót", "rotate"),
            ("Skalowanie", "scale")
        ]

        for text, mode in modes:
            ttk.Radiobutton(control_frame, text=text, variable=self.mode_var,
                            value=mode, command=self.change_mode).pack(anchor=tk.W)

        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        self.draw_frame = ttk.LabelFrame(control_frame, text="Rysowanie")
        self.draw_frame.pack(fill=tk.X, pady=5)

        ttk.Button(self.draw_frame, text="Zakończ figurę",
                   command=self.finish_polygon).pack(pady=5)
        ttk.Button(self.draw_frame, text="Anuluj",
                   command=self.cancel_drawing).pack(pady=5)

        ttk.Label(self.draw_frame, text="Lub wprowadź punkty (x,y):").pack()
        self.points_text = tk.Text(self.draw_frame, height=4, width=25)
        self.points_text.pack(pady=5)
        ttk.Button(self.draw_frame, text="Utwórz figurę z tekstu",
                   command=self.create_polygon_from_text).pack(pady=5)

        self.translate_frame = ttk.LabelFrame(control_frame, text="Przesunięcie")

        ttk.Label(self.translate_frame, text="Wektor (dx, dy):").pack()
        trans_input_frame = ttk.Frame(self.translate_frame)
        trans_input_frame.pack()

        self.dx_entry = ttk.Entry(trans_input_frame, width=10)
        self.dx_entry.pack(side=tk.LEFT, padx=2)
        self.dy_entry = ttk.Entry(trans_input_frame, width=10)
        self.dy_entry.pack(side=tk.LEFT, padx=2)

        ttk.Button(self.translate_frame, text="Przesuń",
                   command=self.translate_by_input).pack(pady=5)
        ttk.Label(self.translate_frame, text="Lub przeciągnij myszą").pack()

        self.rotate_frame = ttk.LabelFrame(control_frame, text="Obrót")

        ttk.Label(self.rotate_frame, text="Punkt obrotu (x, y):").pack()
        pivot_frame = ttk.Frame(self.rotate_frame)
        pivot_frame.pack()

        self.pivot_x_entry = ttk.Entry(pivot_frame, width=10)
        self.pivot_x_entry.pack(side=tk.LEFT, padx=2)
        self.pivot_y_entry = ttk.Entry(pivot_frame, width=10)
        self.pivot_y_entry.pack(side=tk.LEFT, padx=2)

        ttk.Button(self.rotate_frame, text="Ustaw punkt",
                   command=self.set_pivot_from_input).pack(pady=5)

        ttk.Label(self.rotate_frame, text="Kąt (stopnie):").pack()
        self.angle_entry = ttk.Entry(self.rotate_frame, width=15)
        self.angle_entry.pack(pady=2)

        ttk.Button(self.rotate_frame, text="Obróć",
                   command=self.rotate_by_input).pack(pady=5)
        ttk.Label(self.rotate_frame, text="Lub kliknij punkt i obracaj myszą").pack()

        self.scale_frame = ttk.LabelFrame(control_frame, text="Skalowanie")

        ttk.Label(self.scale_frame, text="Punkt skalowania (x, y):").pack()
        scale_pivot_frame = ttk.Frame(self.scale_frame)
        scale_pivot_frame.pack()

        self.scale_pivot_x_entry = ttk.Entry(scale_pivot_frame, width=10)
        self.scale_pivot_x_entry.pack(side=tk.LEFT, padx=2)
        self.scale_pivot_y_entry = ttk.Entry(scale_pivot_frame, width=10)
        self.scale_pivot_y_entry.pack(side=tk.LEFT, padx=2)

        ttk.Button(self.scale_frame, text="Ustaw punkt",
                   command=self.set_scale_pivot_from_input).pack(pady=5)

        ttk.Label(self.scale_frame, text="Współczynnik:").pack()
        self.scale_entry = ttk.Entry(self.scale_frame, width=15)
        self.scale_entry.pack(pady=2)

        ttk.Button(self.scale_frame, text="Skaluj",
                   command=self.scale_by_input).pack(pady=5)
        ttk.Label(self.scale_frame, text="Lub kliknij punkt i przeciągaj myszą").pack()

        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Button(control_frame, text="Usuń zaznaczoną figurę",
                   command=self.delete_selected).pack(pady=5)
        ttk.Button(control_frame, text="Wyczyść wszystko",
                   command=self.clear_all).pack(pady=5)

        self.change_mode()

        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)

    def change_mode(self):
        self.mode = self.mode_var.get()
        self.pivot_point = None
        self.selected_polygon = None
        self.current_points = []

        self.draw_frame.pack_forget()
        self.translate_frame.pack_forget()
        self.rotate_frame.pack_forget()
        self.scale_frame.pack_forget()

        if self.mode == "draw":
            self.draw_frame.pack(fill=tk.X, pady=5)
            self.canvas.config(cursor="cross")
        elif self.mode == "translate":
            self.translate_frame.pack(fill=tk.X, pady=5)
            self.canvas.config(cursor="hand2")
        elif self.mode == "rotate":
            self.rotate_frame.pack(fill=tk.X, pady=5)
            self.canvas.config(cursor="exchange")
        elif self.mode == "scale":
            self.scale_frame.pack(fill=tk.X, pady=5)
            self.canvas.config(cursor="sizing")

        self.redraw_canvas()

    def on_canvas_click(self, event):
        point = Point(event.x, event.y)

        if self.mode == "draw":
            self.current_points.append(point)
            self.redraw_canvas()

        elif self.mode == "translate":
            self.selected_polygon = self.find_polygon_at(point)
            if self.selected_polygon:
                self.drag_start = point
                self.redraw_canvas()

        elif self.mode == "rotate":
            if not self.pivot_point:
                self.pivot_point = point
                self.redraw_canvas()
            else:
                self.selected_polygon = self.find_polygon_at(point)
                if self.selected_polygon:
                    self.drag_start = point
                    self.redraw_canvas()

        elif self.mode == "scale":
            if not self.pivot_point:
                self.pivot_point = point
                self.redraw_canvas()
            else:
                self.selected_polygon = self.find_polygon_at(point)
                if self.selected_polygon:
                    self.drag_start = point
                    self.initial_distance = point.distance_to(self.pivot_point)
                    self.redraw_canvas()

    def on_canvas_drag(self, event):
        if not self.selected_polygon or not self.drag_start:
            return

        current = Point(event.x, event.y)

        if self.mode == "translate":
            h = current.x - self.drag_start.x
            v = current.y - self.drag_start.y
            self.selected_polygon.translate(h, v)
            self.drag_start = current
            self.redraw_canvas()

        elif self.mode == "rotate" and self.pivot_point:
            angle1 = math.atan2(self.drag_start.y - self.pivot_point.y,
                                self.drag_start.x - self.pivot_point.x)
            angle2 = math.atan2(current.y - self.pivot_point.y,
                                current.x - self.pivot_point.x)
            angle_diff = angle2 - angle1
            self.selected_polygon.rotate(self.pivot_point, angle_diff)
            self.drag_start = current
            self.redraw_canvas()

        elif self.mode == "scale" and self.pivot_point:
            current_distance = current.distance_to(self.pivot_point)
            if self.initial_distance > 0:
                k = current_distance / self.initial_distance
                self.selected_polygon.scale(self.pivot_point, k)
                self.initial_distance = current_distance
                self.redraw_canvas()

    def on_canvas_release(self, event):
        self.drag_start = None

    def find_polygon_at(self, point: Point) -> Polygon:
        for polygon in reversed(self.polygons):
            if polygon.contains_point(point, self.canvas):
                return polygon
        return None

    def finish_polygon(self):
        if len(self.current_points) >= 3:
            polygon = Polygon(self.current_points.copy())
            self.polygons.append(polygon)
            self.current_points = []
            self.redraw_canvas()
        else:
            messagebox.showwarning("Uwaga", "Figura musi mieć co najmniej 3 punkty!")

    def cancel_drawing(self):
        self.current_points = []
        self.redraw_canvas()

    def create_polygon_from_text(self):
        text = self.points_text.get("1.0", tk.END).strip()
        if not text:
            return

        try:
            points = []
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                coords = line.replace('(', '').replace(')', '').split(',')
                x, y = float(coords[0].strip()), float(coords[1].strip())
                points.append(Point(x, y))

            if len(points) >= 3:
                polygon = Polygon(points)
                self.polygons.append(polygon)
                self.points_text.delete("1.0", tk.END)
                self.redraw_canvas()
            else:
                messagebox.showwarning("Uwaga", "Podaj co najmniej 3 punkty!")
        except Exception as e:
            messagebox.showerror("Błąd", f"Nieprawidłowy format danych: {str(e)}")

    def translate_by_input(self):
        if not self.selected_polygon:
            messagebox.showwarning("Uwaga", "Najpierw wybierz figurę!")
            return

        try:
            h = float(self.dx_entry.get())
            v = float(self.dy_entry.get())
            self.selected_polygon.translate(h, v)
            self.redraw_canvas()
        except ValueError:
            messagebox.showerror("Błąd", "Wprowadź prawidłowe liczby!")

    def set_pivot_from_input(self):
        try:
            x = float(self.pivot_x_entry.get())
            y = float(self.pivot_y_entry.get())
            self.pivot_point = Point(x, y)
            self.redraw_canvas()
        except ValueError:
            messagebox.showerror("Błąd", "Wprowadź prawidłowe współrzędne!")

    def rotate_by_input(self):
        if not self.selected_polygon:
            messagebox.showwarning("Uwaga", "Najpierw wybierz figurę!")
            return

        if not self.pivot_point:
            messagebox.showwarning("Uwaga", "Najpierw ustaw punkt obrotu!")
            return

        try:
            angle_deg = float(self.angle_entry.get())
            angle_rad = math.radians(angle_deg)
            self.selected_polygon.rotate(self.pivot_point, angle_rad)
            self.redraw_canvas()
        except ValueError:
            messagebox.showerror("Błąd", "Wprowadź prawidłowy kąt!")

    def set_scale_pivot_from_input(self):
        try:
            x = float(self.scale_pivot_x_entry.get())
            y = float(self.scale_pivot_y_entry.get())
            self.pivot_point = Point(x, y)
            self.redraw_canvas()
        except ValueError:
            messagebox.showerror("Błąd", "Wprowadź prawidłowe współrzędne!")

    def scale_by_input(self):
        if not self.selected_polygon:
            messagebox.showwarning("Uwaga", "Najpierw wybierz figurę!")
            return

        if not self.pivot_point:
            messagebox.showwarning("Uwaga", "Najpierw ustaw punkt skalowania!")
            return

        try:
            k = float(self.scale_entry.get())
            self.selected_polygon.scale(self.pivot_point, k)
            self.redraw_canvas()
        except ValueError:
            messagebox.showerror("Błąd", "Wprowadź prawidłowy współczynnik!")

    def delete_selected(self):
        if self.selected_polygon and self.selected_polygon in self.polygons:
            self.polygons.remove(self.selected_polygon)
            self.selected_polygon = None
            self.redraw_canvas()

    def clear_all(self):
        if messagebox.askyesno("Potwierdzenie", "Czy na pewno chcesz usunąć wszystkie figury?"):
            self.polygons = []
            self.current_points = []
            self.selected_polygon = None
            self.pivot_point = None
            self.redraw_canvas()

    def redraw_canvas(self):
        self.canvas.delete("all")

        for polygon in self.polygons:
            if len(polygon.points) >= 3:
                coords = []
                for p in polygon.points:
                    coords.extend([p.x, p.y])

                fill_color = "" if polygon == self.selected_polygon else polygon.color
                outline_color = "red" if polygon == self.selected_polygon else "black"

                polygon.canvas_id = self.canvas.create_polygon(
                    coords,
                    outline=outline_color,
                    fill=fill_color,
                    width=2
                )

        for i, point in enumerate(self.current_points):
            self.canvas.create_oval(point.x - 3, point.y - 3, point.x + 3, point.y + 3,
                                    fill="red", outline="black")
            if i > 0:
                prev = self.current_points[i - 1]
                self.canvas.create_line(prev.x, prev.y, point.x, point.y,
                                        fill="blue", width=2)

        if self.pivot_point:
            self.canvas.create_oval(self.pivot_point.x - 5, self.pivot_point.y - 5,
                                    self.pivot_point.x + 5, self.pivot_point.y + 5,
                                    fill="green", outline="black", width=2)
            self.canvas.create_line(self.pivot_point.x - 8, self.pivot_point.y,
                                    self.pivot_point.x + 8, self.pivot_point.y,
                                    fill="green", width=2)
            self.canvas.create_line(self.pivot_point.x, self.pivot_point.y - 8,
                                    self.pivot_point.x, self.pivot_point.y + 8,
                                    fill="green", width=2)

    def save_to_file(self):
        filename = filedialog.asksaveasfilename(defaultextension=".json",
                                                filetypes=[("JSON files", "*.json")])
        if filename:
            try:
                data = [poly.to_dict() for poly in self.polygons]
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=2)
                messagebox.showinfo("Sukces", "Figury zostały zapisane!")
            except Exception as e:
                messagebox.showerror("Błąd", f"Nie udało się zapisać: {str(e)}")

    def load_from_file(self):
        filename = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if filename:
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)
                self.polygons = [Polygon.from_dict(poly_data) for poly_data in data]
                self.redraw_canvas()
                messagebox.showinfo("Sukces", "Figury zostały wczytane!")
            except Exception as e:
                messagebox.showerror("Błąd", f"Nie udało się wczytać: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = PolygonEditor(root)
    root.mainloop()