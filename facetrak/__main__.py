import logging
import tkinter as tk

from .viz.ui import MainWindow


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    root = tk.Tk()
    root.geometry("960x780")
    MainWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
