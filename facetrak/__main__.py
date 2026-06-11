from .ui import MainWindow
import tkinter as tk


def main():
    root = tk.Tk()
    root.geometry("960x780")
    MainWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
