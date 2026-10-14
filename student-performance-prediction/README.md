# Student Performance Prediction System

Predict student academic performance using attendance, assignment scores, internal assessments, and other academic indicators.

## Machine Learning Concepts Covered

| Concept | Implementation |
|---------|----------------|
| **Data Cleaning** | Remove duplicates, handle missing values, outlier treatment (IQR) |
| **Feature Engineering** | Derived features: internal avg, academic progress, engagement score, at-risk flag |
| **Regression Models** | Linear, Ridge, Random Forest, Gradient Boosting |
| **Model Evaluation** | MAE, RMSE, R² score comparison |

## Project Structure

```
student-performance-prediction/
├── data/                          # Dataset (generated)
├── src/
│   ├── data_cleaning.py           # Data cleaning module
│   ├── feature_engineering.py     # Feature creation & preprocessing
│   ├── model_training.py          # Regression model definitions
│   └── evaluation.py              # Evaluation metrics
├── outputs/                       # Results and plots
├── notebooks/                     # Jupyter notebook walkthrough
├── generate_data.py               # Synthetic data generator
├── main.py                        # End-to-end pipeline
└── requirements.txt
```

## Setup

```bash
cd student-performance-prediction
pip install -r requirements.txt
```

## Usage

**Run the full pipeline:**

```bash
python main.py
```

**Generate dataset only:**

```bash
python generate_data.py
```

**Interactive notebook:**

```bash
jupyter notebook notebooks/student_performance_prediction.ipynb
```

**Streamlit web app:**

```bash
streamlit run app.py
```

The app shows everything on one page when you run:

```bash
streamlit run app.py
```

**Sections on the page:**
1. Prediction dataflow diagram  
2. Student prediction form  
3. Model evaluation charts  
4. Actual vs predicted scatter + student table  
5. Correlation heatmap + insights  

Running `python main.py` also saves the same charts to `outputs/`.

## Dataset Features

| Feature | Description |
|---------|-------------|
| `student_id` | Unique student identifier |
| `name` | Student full name |
| `attendance_pct` | Class attendance percentage |
| `assignment_avg` | Average assignment score |
| `internal_assessment_1/2` | Internal exam scores |
| `midterm_score` | Midterm examination score |
| `study_hours_weekly` | Weekly study hours |
| `extracurricular_hours` | Extracurricular activity hours |
| `previous_gpa` | GPA from previous semester |
| `parental_education` | Parent education level |
| **`final_score`** | **Target variable** |

## Expected Output

The pipeline trains four regression models, compares them using MAE/RMSE/R², saves results to `outputs/model_results.csv`, and generates a comparison chart.
