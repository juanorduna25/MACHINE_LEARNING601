from flask import Flask, render_template, request, url_for
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris, load_diabetes
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Agg')  # Configuración esencial para Flask
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# --- MODELO IRIS ---
# Cargar el dataset Iris
iris = load_iris()
X_iris = iris.data
y_iris = iris.target

# Entrenar el modelo RandomForest para Iris
iris_model = RandomForestClassifier(n_estimators=100, random_state=42)
iris_model.fit(X_iris, y_iris)

# --- MODELO DIABETES ---
# Cargar y entrenar el modelo de diabetes
diabetes = load_diabetes()
x_diabetes = diabetes.data[:, np.newaxis, 2]  # Usamos solo la tercera característica
y_diabetes = diabetes.target

diabetes_model = LinearRegression()
diabetes_model.fit(x_diabetes, y_diabetes)

@app.route('/')
def home():
    return render_template('base.html')

@app.route('/clasificador-flores', methods=['GET', 'POST'])
def iris_prediction():
    prediction = None
    
    if request.method == 'POST':
        # Obtener los valores del formulario
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])
        
        # Crear array con los valores para la predicción
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        # Realizar la predicción
        prediction_num = iris_model.predict(input_data)[0]
        
        # Convertir el número predicho al nombre de la flor
        flower_names = {0: 'Iris Setosa', 1: 'Iris Versicolor', 2: 'Iris Virginica'}
        prediction = flower_names[prediction_num]
    
    return render_template('clasificador-flores.html', prediction=prediction)

@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes_prediction():
    prediction = None
    plot_url = None
    input_value = ''

    # Creamos la gráfica base con los datos reales y la línea de regresión
    plt.figure(figsize=(10, 6))
    
    # Ordenamos los datos para la línea de regresión
    x_sorted = np.sort(x_diabetes, axis=0)
    y_line = diabetes_model.predict(x_sorted)
    
    # 1. Graficar la línea de regresión
    plt.plot(x_sorted, y_line, color='red', linewidth=2, label='Modelo')
    
    # 2. Graficar los datos reales
    plt.scatter(x_diabetes, y_diabetes, color='blue', alpha=0.5, label='Datos reales')
    
    if request.method == 'POST':
        input_value = request.form.get('input_value', '')
        try:
            input_float = float(input_value)
            X_new = np.array([[input_float]])
            prediction = diabetes_model.predict(X_new)[0]
            
            # 3. Graficar la predicción
            plt.scatter(X_new, prediction, color='green', s=100, marker='X', label='Predicción')
            
        except ValueError:
            prediction = "Error: Ingrese un número válido"
        except Exception as e:
            prediction = f"Error: {str(e)}"
    
    # Configuración estética
    plt.xlabel('Característica (Normalizada)')
    plt.ylabel('Progresión de Diabetes')
    plt.title('Regresión Lineal - Dataset Diabetes')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()

    # Guardar imagen en memoria
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=120, bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close()

    return render_template('diabetes.html', 
                         prediction=prediction, 
                         plot_url=plot_url, 
                         input_value=input_value)

@app.route('/mapa_mental')
def mapa_mental():
    return render_template('mapa_mental.html')

if __name__ == '__main__':
    app.run(debug=True)