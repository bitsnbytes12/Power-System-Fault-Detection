from flask import Flask, render_template, request
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)


df = pd.read_csv("dataset/fault_data.csv")


le = LabelEncoder()

df["Fault Type Encoded"] = le.fit_transform(df["Fault Type"])


df = pd.get_dummies(
    df,
    columns=[
        "Weather Condition",
        "Maintenance Status",
        "Component Health"
    ],
    drop_first=True
)


X = df.drop(
    columns=[
        "Fault ID",
        "Fault Type",
        "Fault Location (Latitude, Longitude)",
        "Fault Type Encoded"
    ]
)

y = df["Fault Type Encoded"]

# Save training columns
training_columns = X.columns


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


numerical_cols = X_train.select_dtypes(include=["int64", "float64"]).columns

scaler = StandardScaler()

X_train[numerical_cols] = scaler.fit_transform(
    X_train[numerical_cols]
)

X_test[numerical_cols] = scaler.transform(
    X_test[numerical_cols]
)


model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("Model trained successfully.")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = {
        "Voltage (V)": float(request.form["voltage"]),
        "Current (A)": float(request.form["current"]),
        "Power Load (MW)": float(request.form["power"]),
        "Temperature (°C)": float(request.form["temperature"]),
        "Wind Speed (km/h)": float(request.form["wind"]),
        "Duration of Fault (hrs)": float(request.form["duration"]),
        "Down time (hrs)": float(request.form["downtime"])
    }

    weather = request.form["weather"]
    maintenance = request.form["maintenance"]
    health = request.form["health"]

    input_df = pd.DataFrame([data])

    # Add missing one-hot columns
    for col in training_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Set selected categories
    weather_col = f"Weather Condition_{weather}"
    if weather_col in input_df.columns:
        input_df[weather_col] = 1

    maintenance_col = f"Maintenance Status_{maintenance}"
    if maintenance_col in input_df.columns:
        input_df[maintenance_col] = 1

    health_col = f"Component Health_{health}"
    if health_col in input_df.columns:
        input_df[health_col] = 1

    # Arrange columns exactly like training
    input_df = input_df[training_columns]

    # Scale numeric features
    input_df[numerical_cols] = scaler.transform(
        input_df[numerical_cols]
    )

    prediction = model.predict(input_df)

    fault = le.inverse_transform(prediction)[0]

    return render_template(
        "index.html",
        prediction_text=f"Predicted Fault Type: {fault}"
    )


if __name__ == "__main__":
    app.run(debug=True)