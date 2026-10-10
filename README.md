# 📊 Predicción de Abandono de Empleados en Recursos Humanos (Employee Attrition)

> **Proyecto Final Integrador — Diplomatura en Ciencia de Datos y Análisis Avanzado**  
> **Institución:** Universidad Tecnológica Nacional - Facultad Regional Buenos Aires (UTN BA)  
> **Autores:** Paola Chabay, Gustavo Stampone, MPaz Gazzola Bascougnet, María Marta Caffaro  
> **Fecha:** Octubre de 2026  

---

## 📌 1. Descripción y Caso de Negocio

El abandono voluntario y no planificado de personal (*employee attrition*) genera importantes pérdidas operativas, fuga de conocimiento crítico, sobrecarga laboral en los equipos y elevados costos asociados al reclutamiento, contratación e inducción de nuevos colaboradores.

Este proyecto desarrolla una **solución analítica de Machine Learning supervisado** orientada a la Gerencia de Recursos Humanos y Direcciones de Operaciones para anticipar qué colaboradores presentan mayor probabilidad de renuncia (`Attrition = "Yes"`), permitiendo una intervención temprana y personalizada.

### 🎯 Objetivos y Alcance
* **Objetivo de Machine Learning:** Maximizar la detección de casos reales de renuncia asegurando un umbral de **Recall $\ge 0.75$**, con un piso de discriminación global de **AUC-ROC $\ge 0.80$** y evaluando el **PR-AUC** dada la naturaleza desbalanceada del problema.
* **Traducción Operativa para RRHH:** Convertir las métricas estadísticas en una regla de negocio clara: **¿Qué porcentaje de los empleados que realmente renuncian se detectan al auditar al 20% de la plantilla con mayor riesgo predicho?**
* **KPI de Negocio (Reducción del 15% en rotación):** Al tratarse de un dataset sintético y de corte transversal, se establece como **hipótesis de valor y objetivo aspiracional a validar mediante una prueba A/B** en una etapa de implementación real, sin pretender demostrar causalidad en esta fase de Prueba de Concepto (PoC).

---

## 📂 2. Datos y Calidad de la Información

* **Fuente:** *IBM HR Analytics Employee Attrition & Performance* (Kaggle).
* **Volumen:** 1.470 registros y 35 variables.
* **Distribución de Clases:** 
  * `Attrition = No`: 1.233 empleados (83.88%)
  * `Attrition = Yes`: 237 empleados (16.12%)
* **Limpieza y Consistencia:**
  * Se eliminaron atributos con varianza cero (`EmployeeCount`, `StandardHours`, `Over18`) y el identificador `EmployeeNumber`.
  * **Tratamiento fundamentado de Outliers:** No se descartaron registros atípicos de forma automática por IQR. Se verificó que los 114 outliers de `MonthlyIncome` correspondían a cargos de alta jerarquía (`Manager` y `Research Director`), y que los 104 outliers de `YearsAtCompany` eran lógicamente consistentes con los años totales de carrera (`TotalWorkingYears`). Se conservaron y trataron mediante `RobustScaler` dentro del pipeline para no sesgar el modelo ni perder información de personal senior.

---

## 🛠️ 3. Ingeniería de Características (Feature Engineering)

Para capturar señales de trayectoria, equidad y clima laboral sin generar fuga de datos (*data leakage*), se construyeron 6 variables derivadas (calculadas fila a fila):
1. **`IngresoPorAnioExperiencia`**: `MonthlyIncome / (TotalWorkingYears + 1)`
2. **`RatioAntiguedadEmpresa`**: `YearsAtCompany / (TotalWorkingYears + 1)`
3. **`RatioEstancamientoPromocion`**: `YearsSinceLastPromotion / (YearsAtCompany + 1)`
4. **`RatioAntiguedadConManager`**: `YearsWithCurrManager / (YearsAtCompany + 1)`
5. **`RotacionLaboralPrevia`**: `NumCompaniesWorked / (TotalWorkingYears + 1)`
6. **`IndiceSatisfaccionGeneral`**: Promedio simple de satisfacción laboral, ambiental, relacional y equilibrio vida-trabajo.

---

## 🔬 4. Modelado y Resultados

Se implementó un esquema de validación con división estratificada (80% train / 20% test, con 47 casos positivos en test) y preprocesamiento unificado vía `ColumnTransformer` (`OneHotEncoder` para variables categóricas y `RobustScaler` para numéricas). Se optimizó el umbral de decisión para cumplir con $\text{Recall} \ge 0.75$.

### 📊 Tabla Comparativa de Modelos (Test Set)

| Modelo | Umbral | Recall | Precision | F1-Score | AUC-ROC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Regresión Logística (Ganador)** | **0.373** | **0.766** | **0.324** | **0.456** | **0.808** | **0.558** |
| **Random Forest** | 0.200 | 0.766 | 0.333 | 0.465 | 0.759 | 0.401 |
| **XGBoost** | 0.150 | 0.766 | 0.295 | 0.426 | 0.763 | 0.493 |

*Nota: La tasa base aleatoria de referencia para PR-AUC es de 0.160.*

* **Criterio de Selección:** La **Regresión Logística** superó el piso mínimo fijado ($\text{AUC-ROC} \ge 0.80$) y superó ampliamente a los modelos de ensamble en **PR-AUC (0.558 vs 0.401 y 0.493)**, ofreciendo además una arquitectura más parsimoniosa e interpretable.
* **Validación Cruzada Estratificada ($k=5$):** En entrenamiento, el baseline alcanzó un $\text{AUC-ROC} = 0.831 \pm 0.023$ y $\text{PR-AUC} = 0.625 \pm 0.060$, confirmando su estabilidad estructural.

---

## 📈 5. Impacto de Negocio: Curva de Ganancia Acumulada

Al clasificar a los empleados del conjunto de prueba por orden descendente de probabilidad predicha:
* Al auditar únicamente al **20% de los empleados con mayor riesgo** (59 de 294 personas), el modelo detecta **28 de las 47 renuncias reales (59.6% de Recall en el Top-20%)**.
* Esto casi **triplica la efectividad de una intervención aleatoria** (que solo detectaría el 20%), permitiendo a RRHH anticipar aproximadamente 6 de cada 10 renuncias asignando recursos solo a la quinta parte del personal.

---

## ⚖️ 6. Equidad y Explicabilidad (SHAP)

* **Análisis de Equidad:**
  * **Género:** Paridad adecuada (Recall de 75.0% en mujeres vs 77.4% en varones).
  * **Edad:** Se detectó una brecha sensible: Recall de 88.2% en el segmento 18-29 años frente a 37.5% en mayores de 50 años. Esta disparidad queda identificada como un punto de auditoría ética y monitoreo antes del pase a producción.
* **Explicabilidad con SHAP:**
  * **Factores de riesgo:** Trabajar horas extras (`OverTime = Yes`), viajar frecuentemente por trabajo (`BusinessTravel_Travel_Frequently`) y contar con antecedentes de alta rotación (`RotacionLaboralPrevia`).
  * **Factores de retención:** Estabilidad en el vínculo con el manager (`RatioAntiguedadConManager`), alto nivel de jerarquía e ingresos (`JobLevel`, `MonthlyIncome`) y satisfacción general elevada.

---

## 📁 7. Estructura del Repositorio

```text
├── README.md                                  # Documentación principal
├── Trabajo_Final_Integrador_ver3.ipynb        # Código reproducible de extremo a extremo
└── requirements.txt                           # Dependencias del proyecto
```

---

## 🚀 8. Instalación y Uso

### Clonar el repositorio y configurar el entorno
```bash
# 1. Clonar el repositorio
git clone https://github.com/guss1976/TrabajoFinalCienciaDatos.git
cd TrabajoFinalCienciaDatos

# 2. Crear y activar entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

### Ejecución
* Abrir el notebook en Jupyter o Google Colab:
```bash
jupyter notebook Trabajo_Final_Integrador_ver3.ipynb
```
* Para usar en Google Colab, se puede abrir directamente desde el archivo `.ipynb` y ejecutar las celdas secuencialmente.

---

## 📚 9. Referencias
* Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
* Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *ACM SIGKDD*, 785–794.
* Davis, J., & Goadrich, M. (2006). The relationship between Precision-Recall and ROC curves. *ICML*, 233–240.
* Harter, J. K., et al. (2002). Business-unit-level relationship between employee satisfaction and outcomes. *J. Appl. Psychol.*, 87(2), 268–279.
* Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS 30*, 4765–4774.
