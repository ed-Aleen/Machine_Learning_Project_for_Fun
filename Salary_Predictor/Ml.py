import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import tkinter as tk
from tkinter import simpledialog, messagebox, ttk

# Data preparation
data = pd.read_csv('salary_data.csv')
data.dropna(inplace=True)  # Optionally drop rows with any NaN values

# Encode categorical data
categorical_features = ['Gender', 'Education Level', 'Job Title']
numeric_features = ['Age', 'Years of Experience']

# Imputer for numeric features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('passthrough', 'passthrough')
])

# OneHotEncoder for categorical features
categorical_transformer = OneHotEncoder()

# Create a column transformer to apply different preprocessing to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create a pipeline that applies the preprocessor and then fits a linear model
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', LinearRegression())])

X = data.drop('Salary', axis=1)
y = data['Salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
pipeline.fit(X_train, y_train)

# GUI setup
root = tk.Tk()
root.geometry("400x300")

# Dropdown options extracted from the data (ensure data has no NaN in these columns)
gender_options = sorted(data['Gender'].unique())
education_options = sorted(data['Education Level'].unique())
job_title_options = sorted(data['Job Title'].unique())

# Variables to store dropdown choices
gender_var = tk.StringVar(root)
education_var = tk.StringVar(root)
job_title_var = tk.StringVar(root)
gender_var.set(gender_options[0])
education_var.set(education_options[0])
job_title_var.set(job_title_options[0])

def get_input():
    try:
        age = int(simpledialog.askstring("Input", "Enter age:"))
        years_exp = int(simpledialog.askstring("Input", "Enter years of experience:"))

        # Prepare input for prediction
        input_data = pd.DataFrame([[age, gender_var.get(), education_var.get(), job_title_var.get(), years_exp]],
                                  columns=['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience'])
        
        # Predict salary
        predicted_salary = pipeline.predict(input_data)[0]
        result_label.config(text=f"Predicted Salary: ${predicted_salary:.2f}")
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter valid numeric values for age and years of experience.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")
        result_label.config(text="Error in prediction")

# Dropdown menus for Gender, Education, and Job Title
gender_menu = ttk.OptionMenu(root, gender_var, gender_options[0], *gender_options)
gender_menu.pack(pady=10)
education_menu = ttk.OptionMenu(root, education_var, education_options[0], *education_options)
education_menu.pack(pady=10)
job_title_menu = ttk.OptionMenu(root, job_title_var, job_title_options[0], *job_title_options)
job_title_menu.pack(pady=10)

# Button to trigger input
predict_button = tk.Button(root, text="Enter Data", command=get_input)
predict_button.pack(pady=20)

# Label to display the result
result_label = tk.Label(root, text="Predicted Salary: $0")
result_label.pack(pady=20)

root.mainloop()
