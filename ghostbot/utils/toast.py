# --- UI：暗黑极客风弹窗 ---
def win_toast(text):
    try:
        import tkinter as tk
        root = tk.Tk()
        root.overrideredirect(True)
        root.attributes("-topmost", True)

        bg_color = "#1E1E1E"
        text_color = "#569CD6"
        root.configure(bg=bg_color, highlightthickness=1, highlightbackground="#333333")

        label = tk.Label(
            root, text=f"👻 Ghost Agent\n\n{text}", bg=bg_color, fg=text_color,
            font=("Consolas", 11), justify="left", padx=20, pady=15, wraplength=400
        )
        label.pack()

        root.update_idletasks()
        x = root.winfo_screenwidth() - root.winfo_width() - 30
        y = root.winfo_screenheight() - root.winfo_height() - 60
        root.geometry(f"+{x}+{y}")

        root.after(4000, root.destroy)
        root.mainloop()
    except Exception as e:
        print(f"弹窗渲染失败: {e}")