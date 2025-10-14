# app.py
import os
import io
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from flask import Flask, request, render_template_string, send_file
from sqlalchemy import create_engine
from sklearn.linear_model import LinearRegression
from scipy import stats

app = Flask(__name__)

# SQLite database for logs
engine = create_engine("sqlite:///logs.db")

TEMPLATE = """
<!doctype html>
<title>Data Science Playground</title>
<h2>Upload a CSV file for analysis</h2>
<form method=post enctype=multipart/form-data>
  <input type=file name=file>
  <input type=submit value=Upload>
</form>
{% if message %}
<p>{{ message }}</p>
{% endif %}
{% if plot_url %}
<img src="{{ plot_url }}">
{% endif %}
"""

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            df = pd.read_csv(file)
            
            # Quick stats
            summary = df.describe().to_string()
            print("Summary:\n", summary)
            
            # Save log in DB
            df.head().to_sql("last_upload", engine, if_exists="replace", index=False)
            
            # Simple ML: predict first numeric column with linear regression
            num_cols = df.select_dtypes(include=np.number).columns
            if len(num_cols) >= 2:
                X = df[[num_cols[0]]].fillna(0)
                y = df[num_cols[1]].fillna(0)
                model = LinearRegression().fit(X, y)
                r2 = model.score(X, y)
                message = f"Linear regression {num_cols[0]} -> {num_cols[1]}, R² = {r2:.2f}"
            else:
                message = "Not enough numeric data for regression."
            
            # Visualization with seaborn
            plt.figure(figsize=(6, 4))
            sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
            img = io.BytesIO()
            plt.savefig(img, format="png")
            img.seek(0)
            return send_file(img, mimetype="image/png")
            
            return render_template_string(TEMPLATE, message=message, plot_url="/plot.png")
    return render_template_string(TEMPLATE, message=None)

if __name__ == "__main__":
    app.run(debug=True)
