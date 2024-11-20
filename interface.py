import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import pandas as pd
from main import carregar_dados, treinar_modelo, visualizar_arvore, exibir_matriz_confusao

class WeatherApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Previsão de Chuva")
        self.root.geometry("600x500")
        self.root.configure(bg="#87CEEB") 

        self.X_train, self.X_test, self.y_train, self.y_test = carregar_dados()
        self.clf = treinar_modelo(self.X_train, self.y_train)

        self.sun_icon = ImageTk.PhotoImage(Image.open("imgs/sun.png").resize((100, 100)))
        self.rain_icon = ImageTk.PhotoImage(Image.open("imgs/rain.png").resize((100, 100)))

        self.create_widgets()

    def create_widgets(self):
        def on_enter(e, button, hover_color):
            button.config(bg=hover_color)

        def on_leave(e, button, original_color):
            button.config(bg=original_color)

        title_frame = tk.Frame(self.root, bg="#87CEEB")
        title_frame.pack(pady=10)

        self.logo_icon = ImageTk.PhotoImage(Image.open("imgs/logo.png").resize((50, 50)))
        logo_label = tk.Label(title_frame, image=self.logo_icon, bg="#87CEEB")
        logo_label.pack(side=tk.LEFT, padx=5)

        title = tk.Label(title_frame, text="Previsão de Chuva", font=("Arial", 24, "bold"), bg="#87CEEB", fg="white")
        title.pack(side=tk.LEFT, padx=5)

        btn_arvore = tk.Button(self.root, text="Visualizar Árvore de Decisão", font=("Arial", 12),
                            bg="#1E90FF", fg="white", command=lambda: visualizar_arvore(self.clf, self.X_train))
        btn_arvore.pack(pady=10)
        btn_arvore.bind("<Enter>", lambda e: on_enter(e, btn_arvore, "#4682B4"))
        btn_arvore.bind("<Leave>", lambda e: on_leave(e, btn_arvore, "#1E90FF"))

        btn_matriz = tk.Button(self.root, text="Exibir Matriz de Confusão", font=("Arial", 12),
                            bg="#1E90FF", fg="white", command=lambda: exibir_matriz_confusao(self.clf, self.X_test, self.y_test))
        btn_matriz.pack(pady=10)
        btn_matriz.bind("<Enter>", lambda e: on_enter(e, btn_matriz, "#4682B4"))
        btn_matriz.bind("<Leave>", lambda e: on_leave(e, btn_matriz, "#1E90FF"))

        lbl_input = tk.Label(self.root, text="Inserir Novos Dados para Previsão", font=("Arial", 16, "bold"), bg="#87CEEB", fg="white")
        lbl_input.pack(pady=10)

        self.create_input_fields()

        btn_frame = tk.Frame(self.root, bg="#87CEEB")
        btn_frame.pack(pady=10)

        # Botão Prever
        btn_prever = tk.Button(btn_frame, text="Prever", font=("Arial", 12), bg="#32CD32", fg="white", command=self.prever)
        btn_prever.pack(side=tk.LEFT, padx=5)
        btn_prever.bind("<Enter>", lambda e: on_enter(e, btn_prever, "#228B22"))
        btn_prever.bind("<Leave>", lambda e: on_leave(e, btn_prever, "#32CD32"))

        # Botão Limpar
        btn_clear = tk.Button(btn_frame, text="Limpar", font=("Arial", 12), bg="#FF6347", fg="white", command=self.reset_inputs)
        btn_clear.pack(side=tk.LEFT, padx=5)
        btn_clear.bind("<Enter>", lambda e: on_enter(e, btn_clear, "#CD5C5C"))
        btn_clear.bind("<Leave>", lambda e: on_leave(e, btn_clear, "#FF6347"))

        self.result_frame = tk.Frame(self.root, bg="#87CEEB")
        self.result_frame.pack(pady=10)

        self.result_icon_label = tk.Label(self.result_frame, bg="#87CEEB")
        self.result_icon_label.pack(side=tk.LEFT, padx=10)

        self.result_text_label = tk.Label(self.result_frame, font=("Arial", 16, "bold"), bg="#87CEEB")
        self.result_text_label.pack(side=tk.LEFT, padx=10)

    def create_input_fields(self):
        frame = tk.Frame(self.root, bg="#87CEEB")
        frame.pack(pady=10)

        self.entries = {}
        labels = ["Temperatura (°C)", "Umidade (%)", "Velocidade do Vento (km/h)", "Cobertura de Nuvens (%)", "Pressão Atmosférica (hPa)"]
        for label in labels:
            row = tk.Frame(frame, bg="#87CEEB")
            lbl = tk.Label(row, text=label, width=25, anchor="w", font=("Arial", 10, "bold"), bg="#87CEEB", fg="white")
            entry = tk.Entry(row, width=10, font=("Arial", 10))
            lbl.pack(side=tk.LEFT, padx=5)
            entry.pack(side=tk.RIGHT, padx=5)
            row.pack(fill=tk.X, pady=2)
            self.entries[label] = entry

    def prever(self):
        try:
            data = {
                'Temperature': [float(self.entries["Temperatura (°C)"].get())],
                'Humidity': [float(self.entries["Umidade (%)"].get())],
                'Wind_Speed': [float(self.entries["Velocidade do Vento (km/h)"].get())],
                'Cloud_Cover': [float(self.entries["Cobertura de Nuvens (%)"].get())],
                'Pressure': [float(self.entries["Pressão Atmosférica (hPa)"].get())]
            }
            input_df = pd.DataFrame(data)

            prediction = self.clf.predict(input_df)

            if prediction[0] == 1:
                result = "Alta probabilidade de chuva."
                self.result_icon_label.configure(image=self.rain_icon)
                self.result_text_label.configure(text=result, fg="blue")
            else:
                result = "Baixa probabilidade de chuva."
                self.result_icon_label.configure(image=self.sun_icon)
                self.result_text_label.configure(text=result, fg="orange")
        except ValueError:
            messagebox.showerror("Erro", "Por favor, insira valores numéricos válidos.")

    def reset_inputs(self):
        """Reseta os campos de entrada e o resultado exibido."""
        for entry in self.entries.values():
            entry.delete(0, tk.END)  

        self.result_icon_label.configure(image="")
        self.result_text_label.configure(text="")

if __name__ == "__main__":
    root = tk.Tk()
    app = WeatherApp(root)
    root.mainloop()
