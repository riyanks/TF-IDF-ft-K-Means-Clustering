import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import pandas as pd
from preprocessing import preprocess_text
from feature_extraction import extract_features
from clustering import find_optimal_k
from clustering import show_kmeans

class DataProcessingApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Data Processing App")
        # Frame untuk input
        self.Ky = None
        self.Kx = None

        self.input_frame = ttk.LabelFrame(self.master, text="Insert Data" ,padding=(20, 10))
        self.input_frame.grid(row=0, column=0,  padx=(20, 10), pady=(20, 10), sticky="nsew")

        ttk.Label(self.input_frame, text="Pilih File CSV:").grid(row=0, column=0, sticky="ew", pady=(10,5))
        self.file_path_entry = ttk.Entry(self.input_frame, state="readonly")
        self.file_path_entry.grid(row=1, column=0,sticky="ew", padx=5, pady=(0, 5))

        ttk.Button(self.input_frame, text="Browse", command=self.browse_file).grid(row=1, column=1, padx=5, pady=(0, 5))

        ttk.Label(self.input_frame, text="Pilih Header:").grid(row=3, column=0, sticky="w", pady=(0, 5))
        self.header_combobox = ttk.Combobox(self.input_frame, values=[], state="readonly")
        self.header_combobox.grid(row=4, column=0, padx=5, pady=(0, 10))

        ttk.Button(self.input_frame, text="Proses Data", command=self.process_data, style="Accent.TButton").grid(row=6, column=0, padx=5, pady=10)

        self.pane_2 = ttk.Frame(self.master)
        self.pane_2.grid(row=0, column=1, pady=(25, 5), sticky="nsew", rowspan=3)

        # Notebook
        self.notebook = ttk.Notebook(self.pane_2)

        # Tab #1
        self.tab_1 = ttk.Frame(self.notebook)
        self.tab_1.columnconfigure(index=0, weight=1)
        self.tab_1.columnconfigure(index=1, weight=1)
        self.tab_1.rowconfigure(index=0, weight=1)
        self.tab_1.rowconfigure(index=1, weight=1)
        self.notebook.add(self.tab_1, text="Preprocessing")


        # Label
        self.scroolbar = ttk.Scrollbar(self.tab_1, orient="vertical")
        self.scroolbar.grid(row=0, column=1, sticky="ns")
        self.output_text1 = tk.Text(self.tab_1, wrap="word", width=60, height=10, yscrollcommand=self.scroolbar.set)
        self.output_text1.grid(row=0, column=0, sticky="nsew")

        # Tab #2
        self.tab_2 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_2, text="Nilai LSA")

        self.scroolbar = ttk.Scrollbar(self.tab_2, orient="vertical")
        self.scroolbar.grid(row=0, column=1, sticky="ns")
        self.output_text2 = tk.Text(self.tab_2, wrap="word", width=60, height=10, yscrollcommand=self.scroolbar.set)
        self.output_text2.grid(row=0, column=0, sticky="nsew")

        # Tab #3
        self.tab_3 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_3, text="Nilai K")
        self.scroolbar = ttk.Scrollbar(self.tab_3, orient="vertical")
        self.scroolbar.grid(row=0, column=1, sticky="ns")
        self.output_text3 = tk.Text(self.tab_3, wrap="word", width=60, height=10, yscrollcommand=self.scroolbar.set)
        self.output_text3.grid(row=0, column=0, sticky="nsew")

        self.tab_4 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_4, text="Grafik K")
        ttk.Button(self.tab_4, text="Click Here", command=self.show_graph).grid(row=0, column=0, padx=5, pady=5)


        self.notebook.pack(expand=True, fill="both", padx=5, pady=5)

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.file_path_entry.configure(state="normal")
            self.file_path_entry.delete(0, "end")
            self.file_path_entry.insert(0, file_path)
            self.file_path_entry.configure(state="readonly")

            # Mendapatkan header dari file CSV
            headers = pd.read_csv(file_path, nrows=0).columns.tolist()
            self.header_combobox["values"] = headers
            self.header_combobox.set(headers[0])


    def process_data(self):
        file_path = self.file_path_entry.get()
        selected_header = self.header_combobox.get()

        if not file_path:
            df = pd.read_csv(file_path, nrows=500)
            return

        try:
            df = pd.read_csv(file_path, nrows=500)
        except Exception as e:
            self.show_output(f"Error: {str(e)}")
            return

        if selected_header not in df.columns:
            self.show_output("Header yang dipilih tidak ada dalam file.")
            return

        # Tahap Preprocessing
        df[selected_header] = df[selected_header].apply(preprocess_text)

        # Tahap Ekstraksi Fitur
        X_lsa, explained_variance = extract_features(df[selected_header])

        # Tahap Pencarian K menggunakan Elbow Method
        inertias, K_values = find_optimal_k(X_lsa)
        self.Kx, self.Ky = inertias, K_values

        # Menampilkan hasil
        result_text1 = f"Hasil Preprocessing:\n{df[selected_header].head()}\n\n"
        result_text2 = f"Hasil Ekstraksi Fitur (Explained Variance: {explained_variance * 100:.1f}%):\n{X_lsa}\n\n"
        result_text3 = f"Hasil Pencarian K menggunakan Elbow Method:\n{inertias}"

        self.show_output1(result_text1)
        self.show_output2(result_text2)
        self.show_output3(result_text3)

    def show_output1(self, text):
        self.output_text1.configure(state="normal")
        self.output_text1.delete(1.0, "end")
        self.output_text1.insert("end", text)
        self.output_text1.configure(state="disabled")

    def show_output2(self, text):
        self.output_text2.configure(state="normal")
        self.output_text2.delete(1.0, "end")
        self.output_text2.insert("end", text)
        self.output_text2.configure(state="disabled")


    def show_output3(self, text):
        self.output_text3.configure(state="normal")
        self.output_text3.delete(1.0, "end")
        self.output_text3.insert("end", text)
        self.output_text3.configure(state="disabled")
    
    def show_graph(self):
        show_kmeans(self.Kx, self.Ky)
    


if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style(root)
    root.tk.call("source", "forest-light.tcl")
    root.tk.call("source", "forest-dark.tcl")
    style.theme_use("forest-dark")
    app = DataProcessingApp(root)
    root.mainloop()
