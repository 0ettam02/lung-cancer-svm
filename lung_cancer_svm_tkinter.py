import tkinter as tk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split

# Funzione per separare sezioni di testo
def divisione():
    return "-" * 20

# Caricamento del dataset
dataset = pd.read_csv("survey lung cancer.csv")

# PREPARAZIONE DATI
df_one_hot = pd.get_dummies(dataset, columns=['GENDER', "LUNG_CANCER"])
divisione()

X = df_one_hot.drop(columns=['LUNG_CANCER_NO', 'LUNG_CANCER_YES'])
y = df_one_hot['LUNG_CANCER_YES']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

sc = StandardScaler()
sc.fit(X_train)
X_train_sd = sc.transform(X_train)
X_test_sd = sc.transform(X_test)

# Grid Search per SVM
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'degree': [2, 3, 4],
    'gamma': ['scale', 'auto']
}

grid_search = GridSearchCV(estimator=SVC(), param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_sd, y_train)
best_params = grid_search.best_params_

# Addestramento del modello migliore
best_model = SVC(**best_params)
best_model.fit(X_train_sd, y_train)

# Funzione per preprocessare l'input
def preprocess_input(new_data):
    new_data_one_hot = pd.get_dummies(new_data, columns=['GENDER'])
    new_data_one_hot = new_data_one_hot.reindex(columns=X.columns, fill_value=0)
    new_data_scaled = sc.transform(new_data_one_hot)
    return new_data_scaled

# Creazione dell'interfaccia grafica
class LungCancerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Previsione Tumore ai Polmoni")
        self.geometry("800x600")
        
        # Frame per i campi di input
        input_frame = tk.Frame(self)
        input_frame.pack(pady=10)
        
        # Input fields
        self.entries = {}
        fields = ['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 
                  'CHRONIC DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING', 'ALCOHOL CONSUMING', 
                  'COUGHING', 'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN']
        
        for field in fields:
            row = tk.Frame(input_frame)
            label = tk.Label(row, width=22, text=field, anchor='w')
            entry = tk.Entry(row)
            row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
            label.pack(side=tk.LEFT)
            entry.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
            self.entries[field] = entry
        
        # Pulsante per la previsione
        predict_button = tk.Button(self, text="Prevedi", command=self.predict)
        predict_button.pack(pady=10)
        
        # Canvas per il grafico
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    
    def predict(self):
        try:
            nome = "Utente"
            dati = {
                'GENDER': [self.entries['GENDER'].get()],
                'AGE': [int(self.entries['AGE'].get())],
                'SMOKING': [int(self.entries['SMOKING'].get())],
                'YELLOW_FINGERS': [int(self.entries['YELLOW_FINGERS'].get())],
                'ANXIETY': [int(self.entries['ANXIETY'].get())],
                'PEER_PRESSURE': [int(self.entries['PEER_PRESSURE'].get())],
                'CHRONIC DISEASE': [int(self.entries['CHRONIC DISEASE'].get())],
                'FATIGUE': [int(self.entries['FATIGUE'].get())],
                'ALLERGY': [int(self.entries['ALLERGY'].get())],
                'WHEEZING': [int(self.entries['WHEEZING'].get())],
                'ALCOHOL CONSUMING': [int(self.entries['ALCOHOL CONSUMING'].get())],
                'COUGHING': [int(self.entries['COUGHING'].get())],
                'SHORTNESS OF BREATH': [int(self.entries['SHORTNESS OF BREATH'].get())],
                'SWALLOWING DIFFICULTY': [int(self.entries['SWALLOWING DIFFICULTY'].get())],
                'CHEST PAIN': [int(self.entries['CHEST PAIN'].get())]
            }

            new_data = pd.DataFrame(dati)
            new_data_scaled = preprocess_input(new_data)
            new_predict = best_model.predict(new_data_scaled)

            if new_predict[0] == 0:
                messagebox.showinfo("Risultato", f"{nome} non è propenso ad avere un tumore ai polmoni")
            else:
                messagebox.showinfo("Risultato", f"{nome} è propenso ad avere un tumore ai polmoni")

            self.plot_graph()

        except ValueError as e:
            messagebox.showerror("Errore", f"Input non valido: {e}")
    
    def plot_graph(self):
        feature_to_plot = 0 
        probabilities = best_model.decision_function(X_test_sd)
        self.ax.clear()
        self.ax.scatter(X_test_sd[:, feature_to_plot], probabilities, color='green', label='Probabilità Previste (Test)')
        self.ax.scatter(X_train_sd[:, feature_to_plot], y_train, color='blue', label='Dati di Addestramento')
        self.ax.axhline(0, color='red', linestyle='dashed', linewidth=1, label='Soglia di Decisione')
        self.ax.set_xlabel('Età Standardizzata')
        self.ax.set_ylabel('Funzione di Decisione')
        self.ax.set_title('SVM: Funzione di Decisione vs Età')
        self.ax.legend()
        self.canvas.draw()

if __name__ == "__main__":
    app = LungCancerApp()
    app.mainloop()
