import tkinter as tk


def on_button_click():
    label.config(text="Hello, World!")


root = tk.Tk()
root.title("My App")

label = tk.Label(root, text="Welcome to My App")
label.pack(pady=20)

button = tk.Button(root, text="Click Me", command=on_button_click)
button.pack(pady=10)

root.mainloop()
