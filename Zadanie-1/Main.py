from abc import ABC, abstractmethod
from tkinter import *
from tkinter import ttk, filedialog, messagebox
import json


class Shape(ABC):
    def __init__(self):
        self.id = None
        self.selected = False
        self.color = "black"

    @abstractmethod
    def draw(self, canvas):
        pass

    @abstractmethod
    def move(self, dx, dy):
        pass

    @abstractmethod
    def getBounds(self):
        pass

    @abstractmethod
    def getControlPoints(self):
        pass

    @abstractmethod
    def toDict(self):
        pass

    @staticmethod
    @abstractmethod
    def fromDict(data):
        pass

    @abstractmethod
    def containsPoint(self, x, y):
        pass

    def getControlPointAt(self, x, y, tolerance=10):
        for i, (px, py) in enumerate(self.getControlPoints()):
            if abs(px - x) <= tolerance and abs(py - y) <= tolerance:
                return i
        return None


class Line(Shape):
    def __init__(self, x1, y1, x2, y2):
        super().__init__()
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.width = 2

    def draw(self, canvas):
        outline_color = "blue" if self.selected else self.color

        self.id = canvas.create_line(
            self.x1, self.y1, self.x2, self.y2,
            fill=outline_color,
            width=self.width)

        if self.selected:
            self.drawControlPoints(canvas)

    def drawControlPoints(self, canvas):
        radius = 5
        for px, py in self.getControlPoints():
            canvas.create_oval(
                px - radius, py - radius,
                px + radius, py + radius,
                fill="red", outline="red"
            )

    def move(self, dx, dy):
        self.x1 += dx
        self.x2 += dx
        self.y1 += dy
        self.y2 += dy

    def getBounds(self):
        return (
            min(self.x1, self.x2),
            max(self.x1, self.x2),
            min(self.y1, self.y2),
            max(self.y1, self.y2),
        )

    def containsPoint(self, x, y):
        bounds = self.getBounds()
        return bounds[0] <= x <= bounds[1] and bounds[2] <= y <= bounds[3]

    def getControlPoints(self):
        return [(self.x1, self.y1), (self.x2, self.y2)]

    def updateControlPoint(self, index, x, y):
        if index == 0:
            self.x1, self.y1 = x, y
        elif index == 1:
            self.x2, self.y2 = x, y

    def toDict(self):
        return {
            "type": "Line",
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "width": self.width,
            "color": self.color
        }

    @staticmethod
    def fromDict(data):
        line = Line(data["x1"], data["y1"], data["x2"], data["y2"])
        line.width = data.get("width", 2)
        line.color = data.get("color", "black")
        return line


class Rectangle(Shape):
    def __init__(self, x, y, width, height):
        super().__init__()
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.fill_color = ""

    def draw(self, canvas):
        outline_color = "blue" if self.selected else self.color
        outline_width = 2

        self.id = canvas.create_rectangle(
            self.x, self.y, self.x + self.width, self.y + self.height,
            outline=outline_color,
            fill=self.fill_color,
            width=outline_width
        )

        if self.selected:
            self.drawControlPoints(canvas)

    def drawControlPoints(self, canvas):
        radius = 5
        for px, py in self.getControlPoints():
            canvas.create_oval(
                px - radius, py - radius,
                px + radius, py + radius,
                fill="red", outline="red"
            )

    def move(self, dx, dy):
        self.x += dx
        self.y += dy

    def getBounds(self):
        return (self.x, self.x + self.width, self.y, self.y + self.height)


    def containsPoint(self, x, y):
        bounds = self.getBounds()
        return bounds[0] <= x <= bounds[1] and bounds[2] <= y <= bounds[3]


    def getControlPoints(self):
        return [
            (self.x, self.y),
            (self.x + self.width, self.y),
            (self.x + self.width, self.y + self.height),
            (self.x, self.y + self.height)
        ]

    def updateControlPoint(self, index, x, y):
        if index == 0: # lewy górny
            self.width += self.x - x
            self.height += self.y - y
            self.x, self.y = x, y
        elif index == 1: # prawy górny
            self.width = x - self.x
            self.height += self.y - y
            self.y = y
        elif index == 2: # prawy dolny
            self.width = x - self.x
            self.height = y - self.y
        elif index == 3: # lewy dolny
            self.width += self.x - x
            self.x = x
            self.height = y - self.y

    def toDict(self):
        return {
            "type": "Rectangle",
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "color": self.color,
            "fill_color": self.fill_color
        }

    @staticmethod
    def fromDict(data):
        rect = Rectangle(data["x"], data["y"], data["width"], data["height"])
        rect.color = data.get("color", "black")
        rect.fill_color = data.get("fill_color", "")
        return rect


class Circle(Shape):
    def __init__(self, x, y, radius):
        super().__init__()
        self.x = x
        self.y = y
        self.radius = radius
        self.fill_color = ""

    def draw(self, canvas):
        outline_color = "blue" if self.selected else self.color
        self.id = canvas.create_oval(
            self.x - self.radius,
            self.y - self.radius,
            self.x + self.radius,
            self.y + self.radius,
            fill=self.fill_color,
            outline=outline_color,
            width=2
        )

        if self.selected:
            self.drawControlPoints(canvas)

    def drawControlPoints(self, canvas):
        radius = 5
        for px, py in self.getControlPoints():
            canvas.create_oval(
                px - radius, py - radius,
                px + radius, py + radius,
                fill="red", outline="red"
            )

    def getBounds(self):
        return (
            self.x - self.radius,
            self.x + self.radius,
            self.y - self.radius,
            self.y + self.radius
        )

    def containsPoint(self, x, y):
        distance = ((self.x - x) ** 2 + (self.y - y) ** 2) ** 0.5
        return distance <= self.radius

    def move(self, dx, dy):
        self.x += dx
        self.y += dy

    def getControlPoints(self):
        return [
            (self.x, self.y - self.radius),  # góra
            (self.x + self.radius, self.y),  # prawo
            (self.x, self.y + self.radius),  # dół
            (self.x - self.radius, self.y)  # lewo
        ]

    def updateControlPoint(self, index, x, y):
        if index == 0:  # góra
            self.radius = abs(self.y - y)
        elif index == 1:  # prawo
            self.radius = abs(x - self.x)
        elif index == 2:  # dół
            self.radius = abs(y - self.y)
        elif index == 3:  # lewo
            self.radius = abs(self.x - x)

    def toDict(self):
        return {
            "type": "Circle",
            "x": self.x,
            "y": self.y,
            "radius": self.radius,
            "color": self.color,
            "fill_color": self.fill_color
        }

    @staticmethod
    def fromDict(data):
        circle = Circle(data["x"], data["y"], data["radius"])
        circle.color = data.get("color", "black")
        circle.fill_color = data.get("fill_color", "")
        return circle


class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Edytor Kształtów")

        self.shapes = []
        self.selected_shape = None
        self.drawing_mode = None
        self.temp_points = []
        self.drag_data = {"x": 0, "y": 0, "item": None, "control_point": None}

        self.setupUI()
        self.bindEvents()

    def setupUI(self):
        toolbar = Frame(self.root)
        toolbar.pack(side=TOP, fill=X, padx=5, pady=5)

        Label(toolbar, text="Tryb:").pack(side=LEFT, padx=5)

        self.mode_var = StringVar(value="select")
        modes = [
            ("Zaznacz", "select"),
            ("Linia", "line"),
            ("Prostokąt", "rectangle"),
            ("Okrąg", "circle")
        ]

        for text, mode in modes:
            Radiobutton(toolbar, text=text, variable=self.mode_var,
                        value=mode, command=self.changeMode).pack(side=LEFT)

        ttk.Separator(toolbar, orient=VERTICAL).pack(side=LEFT, fill=Y, padx=10)

        Button(toolbar, text="Zapisz", command=self.saveToFile).pack(side=LEFT, padx=5)
        Button(toolbar, text="Wczytaj", command=self.loadFromFile).pack(side=LEFT, padx=5)
        Button(toolbar, text="Wyczyść", command=self.clearCanvas).pack(side=LEFT, padx=5)

        side_panel = Frame(self.root, width=250, bg="lightgray")
        side_panel.pack(side=RIGHT, fill=Y, padx=5, pady=5)
        side_panel.pack_propagate(False)

        Label(side_panel, text="Parametry kształtu", bg="lightgray",
              font=("Arial", 12, "bold")).pack(pady=10)

        self.param_frame = Frame(side_panel, bg="lightgray")
        self.param_frame.pack(fill=BOTH, expand=True, padx=10)

        self.param_entries = {}

        params = ["x1", "y1", "x2", "y2", "x", "y", "width", "height", "radius"]
        for param in params:
            frame = Frame(self.param_frame, bg="lightgray")
            frame.pack(fill=X, pady=2)
            Label(frame, text=f"{param}:", bg="lightgray", width=8, anchor=W).pack(side=LEFT)
            entry = Entry(frame, width=15)
            entry.pack(side=LEFT, padx=5)
            self.param_entries[param] = entry

        Button(side_panel, text="Utwórz kształt",
               command=self.createFromParams).pack(pady=10)
        Button(side_panel, text="Aktualizuj zaznaczony",
               command=self.updateFromParams).pack(pady=5)

        self.canvas = Canvas(self.root, bg="white", width=800, height=600)
        self.canvas.pack(side=LEFT, fill=BOTH, expand=True, padx=5, pady=5)

    def bindEvents(self):
        self.canvas.bind("<Button-1>", self.onCanvasClick)
        self.canvas.bind("<B1-Motion>", self.onCanvasDrag)
        self.canvas.bind("<ButtonRelease-1>", self.onCanvasRelease)

    def changeMode(self):
        self.drawing_mode = self.mode_var.get()
        self.temp_points = []
        if self.selected_shape:
            self.selected_shape.id = None
            self.selected_shape.selected = False
            self.selected_shape = None
            self.clearParamFields()
        self.redraw()

    def onCanvasClick(self, event):
        x, y = event.x, event.y
        mode = self.mode_var.get()

        if mode == "select":
            if self.selected_shape:
                cp_index = self.selected_shape.getControlPointAt(x, y)
                if cp_index is not None:
                    self.drag_data["control_point"] = cp_index
                    self.drag_data["x"] = x
                    self.drag_data["y"] = y
                    return

            clicked_shape = None
            for shape in reversed(self.shapes):
                if shape.containsPoint(x, y):
                    clicked_shape = shape
                    break

            if self.selected_shape:
                self.selected_shape.selected = False

            if clicked_shape:
                clicked_shape.selected = True
                self.selected_shape = clicked_shape
                self.drag_data["item"] = clicked_shape
                self.drag_data["x"] = x
                self.drag_data["y"] = y
                self.loadParamsToFields()
            else:
                self.selected_shape = None
                self.clearParamFields()

            self.redraw()

        elif mode in ["line", "rectangle", "circle"]:
            self.temp_points.append((x, y))

            if mode == "line" and len(self.temp_points) == 2:
                shape = Line(self.temp_points[0][0], self.temp_points[0][1],
                             self.temp_points[1][0], self.temp_points[1][1])
                self.shapes.append(shape)
                self.temp_points = []
                self.redraw()

            elif mode == "rectangle" and len(self.temp_points) == 2:
                x1, y1 = self.temp_points[0]
                x2, y2 = self.temp_points[1]
                shape = Rectangle(min(x1, x2), min(y1, y2),
                                  abs(x2 - x1), abs(y2 - y1))
                self.shapes.append(shape)
                self.temp_points = []
                self.redraw()

            elif mode == "circle" and len(self.temp_points) == 2:
                cx, cy = self.temp_points[0]
                px, py = self.temp_points[1]
                radius = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
                shape = Circle(cx, cy, int(radius))
                self.shapes.append(shape)
                self.temp_points = []
                self.redraw()

    def onCanvasDrag(self, event):
        if self.mode_var.get() != "select":
            return

        x = event.x
        y = event.y

        dx = event.x - self.drag_data["x"]
        dy = event.y - self.drag_data["y"]

        if self.drag_data["control_point"] is not None and self.selected_shape:
            self.selected_shape.updateControlPoint(
                self.drag_data["control_point"],
                x, y
            )
            self.redraw()
            self.loadParamsToFields()

        elif self.drag_data["item"]:
            self.drag_data["item"].move(dx, dy)
            self.drag_data["x"] = x
            self.drag_data["y"] = y
            self.redraw()
            self.loadParamsToFields()

    def onCanvasRelease(self, event):
        self.drag_data = {"x": 0, "y": 0, "item": None, "control_point": None}

    def createFromParams(self):
        mode = self.mode_var.get()

        try:
            if mode == "line":
                x1 = float(self.param_entries["x1"].get())
                y1 = float(self.param_entries["y1"].get())
                x2 = float(self.param_entries["x2"].get())
                y2 = float(self.param_entries["y2"].get())
                self.shapes.append(Line(x1, y1, x2, y2))

            elif mode == "rectangle":
                x = float(self.param_entries["x"].get())
                y = float(self.param_entries["y"].get())
                width = float(self.param_entries["width"].get())
                height = float(self.param_entries["height"].get())
                self.shapes.append(Rectangle(x, y, width, height))

            elif mode == "circle":
                x = float(self.param_entries["x"].get())
                y = float(self.param_entries["y"].get())
                radius = float(self.param_entries["radius"].get())
                self.shapes.append(Circle(x, y, radius))

            self.redraw()
            messagebox.showinfo("Sukces", "Kształt utworzony!")
        except ValueError:
            messagebox.showerror("Błąd", "Wprowadź poprawne wartości liczbowe!")

    def updateFromParams(self):
        if not self.selected_shape:
            messagebox.showwarning("Uwaga", "Nie zaznaczono kształtu!")
            return

        try:
            if isinstance(self.selected_shape, Line):
                self.selected_shape.x1 = float(self.param_entries["x1"].get())
                self.selected_shape.y1 = float(self.param_entries["y1"].get())
                self.selected_shape.x2 = float(self.param_entries["x2"].get())
                self.selected_shape.y2 = float(self.param_entries["y2"].get())

            elif isinstance(self.selected_shape, Rectangle):
                self.selected_shape.x = float(self.param_entries["x"].get())
                self.selected_shape.y = float(self.param_entries["y"].get())
                self.selected_shape.width = float(self.param_entries["width"].get())
                self.selected_shape.height = float(self.param_entries["height"].get())

            elif isinstance(self.selected_shape, Circle):
                self.selected_shape.x = float(self.param_entries["x"].get())
                self.selected_shape.y = float(self.param_entries["y"].get())
                self.selected_shape.radius = float(self.param_entries["radius"].get())

            self.redraw()
            messagebox.showinfo("Sukces", "Kształt zaktualizowany!")
        except ValueError:
            messagebox.showerror("Błąd", "Wprowadź poprawne wartości liczbowe!")

    def loadParamsToFields(self):
        self.clearParamFields()

        if not self.selected_shape:
            return

        if isinstance(self.selected_shape, Line):
            self.param_entries["x1"].insert(0, str(self.selected_shape.x1))
            self.param_entries["y1"].insert(0, str(self.selected_shape.y1))
            self.param_entries["x2"].insert(0, str(self.selected_shape.x2))
            self.param_entries["y2"].insert(0, str(self.selected_shape.y2))

        elif isinstance(self.selected_shape, Rectangle):
            self.param_entries["x"].insert(0, str(self.selected_shape.x))
            self.param_entries["y"].insert(0, str(self.selected_shape.y))
            self.param_entries["width"].insert(0, str(abs(self.selected_shape.width)))
            self.param_entries["height"].insert(0, str(abs(self.selected_shape.height)))

        elif isinstance(self.selected_shape, Circle):
            self.param_entries["x"].insert(0, str(self.selected_shape.x))
            self.param_entries["y"].insert(0, str(self.selected_shape.y))
            self.param_entries["radius"].insert(0, str(self.selected_shape.radius))


    def redraw(self):
        self.canvas.delete("all")
        for shape in self.shapes:
            shape.draw(self.canvas)

    def saveToFile(self):
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if filename:
            data = [shape.toDict() for shape in self.shapes]
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            messagebox.showinfo("Sukces", "Zapisano do pliku!")

    def loadFromFile(self):
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if filename:
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)

                self.shapes = []
                for item in data:
                    shape_type = item["type"]
                    if shape_type == "Line":
                        self.shapes.append(Line.fromDict(item))
                    elif shape_type == "Rectangle":
                        self.shapes.append(Rectangle.fromDict(item))
                    elif shape_type == "Circle":
                        self.shapes.append(Circle.fromDict(item))

                self.redraw()
                messagebox.showinfo("Sukces", "Wczytano z pliku!")
            except Exception as e:
                messagebox.showerror("Błąd", f"Nie udało się wczytać pliku: {e}")

    def clearCanvas(self):
        if messagebox.askyesno("Potwierdzenie", "Czy na pewno wyczyścić canvas?"):
            self.shapes = []
            self.selected_shape = None
            self.redraw()
            self.clearParamFields()

    def clearParamFields(self):
        for entry in self.param_entries.values():
            entry.delete(0, END)


if __name__ == "__main__":
    root = Tk()
    app = DrawingApp(root)
    root.mainloop()