# import streamlit as st 
# import pandas as pd
# import joblib 

# df = pd.read_csv(r"C:\Users\Yash Ohol\OneDrive\Autoprice\data\cars_dataset.csv")
# st.title('🚗 AutoPrice')
# st.write('Predict the price of a used car using Machine Learning. ')

# year = st.number_input('Year', min_value=1996, max_value=2020, value=2019)

# mileage = st.number_input("Mileage", min_value=0, value=30000)

# tax = st.number_input("Tax", min_value=0.0, value=145.0)

# mpg = st.number_input("MPG", min_value=0.0, value=50.0)

# engine_size = st.number_input("Engine Size", min_value=0.0, value=1.6)

# transmission = st.selectbox(
#     "Transmission",
#     ["Manual", "Automatic", "Semi-Auto"]
# )

# fuelType = st.selectbox(
#     'FuelType', 
#     ['Petrol', 'Diesel', 'Hybrid', 'Electric'])

# make = st.selectbox(
#     "Make",
#     sorted(df["Make"].unique())
# )

# available_models = sorted(
#     df[df["Make"] == make]["model"].unique()
# )

# model_name = st.selectbox(
#     "Model",
#     available_models
# )

# model_name = st.selectbox(
#     "Model",
#     ["A1", "A3", "A4", "3 Series", "5 Series"]
# )
# new_car = pd.DataFrame({
#     'year': [year],
#     'mileage': [mileage],
#     'tax': [tax],
#     'mpg': [mpg],
#     'engineSize': [engine_size],
#     'transmission': [transmission],
#     'fuelType': [fuelType],
#     'Make': [make],
#     'model': [model_name]
# })
# model = joblib.load("models/linear_regression_model.pkl")
# model_columns = joblib.load("models/model_columns.pkl")
# new_car_encoded = pd.get_dummies(
#     new_car,
#     columns=['model', 'transmission', 'fuelType', 'Make']
# )
# new_car_encoded = new_car_encoded.reindex(
#     columns=model_columns,
#     fill_value=0
# )

# if st.button("Predict Price"):
#     prediction = model.predict(new_car_encoded)

#     st.success(f"Estimated Car Price: ₹{prediction[0]:,.2f}")
import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AutoPrice",
    page_icon="🚗",
    layout="wide"
)


# --------------------------------------------------
# LOAD DATASET AND MODEL
# --------------------------------------------------

df = pd.read_csv("data/cars_dataset.csv")

model = joblib.load("models/linear_regression_model.pkl")



# Create the same dummy variables that were used
# while training our Linear Regression model

df_encoded = pd.get_dummies(
    df,
    columns=["transmission", "fuelType", "Make", "model"]
)

X = df_encoded.drop(columns=["price"])


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🚗 AutoPrice")

st.subheader("Car Price Prediction System")

st.write(
    "Enter the details of a used car and AutoPrice "
    "will estimate its market price using Linear Regression."
)

st.divider()


# --------------------------------------------------
# CAR DETAILS
# --------------------------------------------------

st.header("🔧 Enter Car Details")


# First row
col1, col2 = st.columns(2)

with col1:

    year = st.number_input(
        "📅 Year",
        min_value=1990,
        max_value=2026,
        value=2019,
        step=1
    )

with col2:

    mileage = st.number_input(
        "🛣️ Mileage",
        min_value=0,
        max_value=400000,
        value=30000,
        step=1000
    )


# Second row
col1, col2 = st.columns(2)

with col1:

    tax = st.number_input(
        "💰 Tax",
        min_value=0.0,
        max_value=1000.0,
        value=145.0,
        step=5.0
    )

with col2:

    mpg = st.number_input(
        "⛽ MPG",
        min_value=0.0,
        max_value=500.0,
        value=50.0,
        step=0.5
    )


# Third row
col1, col2 = st.columns(2)

with col1:

    engine_size = st.number_input(
        "⚙️ Engine Size",
        min_value=0.0,
        max_value=10.0,
        value=1.6,
        step=0.1
    )

with col2:

    transmission = st.selectbox(
        "⚙️ Transmission",
        sorted(df["transmission"].unique())
    )


# Fourth row
col1, col2 = st.columns(2)

with col1:

    fuel_type = st.selectbox(
        "⛽ Fuel Type",
        sorted(df["fuelType"].unique())
    )

with col2:

    make = st.selectbox(
        "🏭 Make",
        sorted(df["Make"].unique())
    )


# --------------------------------------------------
# MODEL DROPDOWN DEPENDS ON MAKE
# --------------------------------------------------

available_models = sorted(
    df[df["Make"] == make]["model"].unique()
)

model_name = st.selectbox(
    "🚘 Model",
    available_models
)


st.divider()



predict_button = st.button(
    "🔮 Predict Car Price",
    type="primary",
    use_container_width=True
)


if predict_button:

    # Create a DataFrame containing the user's input

    new_car = pd.DataFrame({
        "year": [year],
        "mileage": [mileage],
        "tax": [tax],
        "mpg": [mpg],
        "engineSize": [engine_size],
        "transmission": [transmission],
        "fuelType": [fuel_type],
        "Make": [make],
        "model": [model_name]
    })



    new_car_encoded = pd.get_dummies(
        new_car,
        columns=[
            "transmission",
            "fuelType",
            "Make",
            "model"
        ]
    )




    new_car_encoded = new_car_encoded.reindex(
        columns=X.columns,
        fill_value=0
    )


    # Predict price

    prediction = model.predict(new_car_encoded)


 
    predicted_price = prediction[0]

    st.success(
        f"### 💰 Estimated Car Price: ₹{predicted_price:,.0f}"
    )


# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

st.divider()

st.header("📊 Model Performance")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "R² Score",
        "86.7%"
    )

with col2:

    st.metric(
        "MAE",
        "₹2,110"
    )

with col3:

    st.metric(
        "RMSE",
        "₹3,438"
    )


# --------------------------------------------------
# ABOUT THE PROJECT
# --------------------------------------------------

st.divider()

st.header("ℹ️ About AutoPrice")

st.write(
    """
    AutoPrice is a Machine Learning project that uses
    Linear Regression to estimate used-car prices.

    The model was trained using features such as:

    • Year
    • Mileage
    • Tax
    • MPG
    • Engine Size
    • Transmission
    • Fuel Type
    • Make
    • Model
    """
)
