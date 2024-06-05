from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

@app.route("/")
def hello():
    return render_template("index.html")

@app.route("/feedback")
def hello3():
    return render_template("feedback.html")

@app.route("/Diabetes")
def hello4():
    return render_template("diabetes.html")

@app.route("/sub", methods=['POST'])
def submit():
    if request.method == "POST":
        # Fetching form data
        age = float(request.form["age"])
        gender = request.form["gender"]
        pregnant = request.form.get("pregnant", "no")
        months = str(request.form.get("months", "0"))
        glucose = float(request.form["glucose"])
        bloodpressure = float(request.form["bloodpressure"])
        SkinThickness = float(request.form["SkinThickness"])
        insulin = float(request.form["insulin"])
        BMI = float(request.form["BMI"])
        DiabetesPedigreeFunction = float(request.form["DiabetesPedigreeFunction"])

        # Load the model and scaler
        df = pd.read_csv("shuffled_diabetes.csv")
        x = df.iloc[:, 0:8].values
        y = df.iloc[:, 8].values

        sc = StandardScaler()
        x_train = sc.fit_transform(x)
        y_train = y

        # Train the model
        forest = RandomForestClassifier(random_state=0)
        forest.fit(x_train, y_train)

        # Predict on user input
        prediction = forest.predict(sc.transform([[age, glucose, bloodpressure, SkinThickness, insulin, BMI, DiabetesPedigreeFunction, 1 if gender=="male" else 0]]))

        # Convert prediction to "Yes" or "No"
        prediction_result = "Yes" if prediction[0] == 1 else "No"

        # Display prediction
        return render_template("sub.html", age=age, glucose=glucose, c1=bloodpressure, d1=SkinThickness, e1=insulin, f1=BMI, g1=DiabetesPedigreeFunction, n=prediction_result, pregnant=pregnant, months=months)

if __name__ == "__main__":
    app.run(debug=True)
