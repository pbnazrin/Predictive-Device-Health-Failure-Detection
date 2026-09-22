import joblib
import pandas as pd


MODEL_PATH = "models/device_failure_model.joblib"
THRESHOLD_PATH = "models/failure_threshold.joblib"


model = joblib.load(MODEL_PATH)
threshold = joblib.load(THRESHOLD_PATH)


def predict_failure(device_data):
    """
    Predict whether a device is likely to fail within 7 days.

    Parameters
    ----------
    device_data : dict
        Raw device telemetry data.

    Returns
    -------
    dict
        Failure probability and prediction.
    """

    data = pd.DataFrame([device_data])

    probability = model.predict_proba(data)[0, 1]

    prediction = int(probability >= threshold)

    return {
        "failure_probability": round(float(probability), 4),
        "failure_within_7_days": prediction
    }


if __name__ == "__main__":

    sample_device = {
        "OS": "Android",
        "OS_Version": 13.0,
        "Manufacturer": "Samsung",
        "Model": "Model_E",
        "Network_Type": "4G",
        "Battery_Health": 45,
        "Battery_Temperature": 44,
        "CPU_Usage": 85,
        "Memory_Usage": 80,
        "Storage_Usage": 70,
        "Network_Signal": 35,
        "App_Crash_Count": 5,
        "Reboot_Count": 4,
        "Error_Count": 6,
        "Data_Usage": 2500,
        "Days_Since_Last_Update": 100,
        "Last_Checkin_Hours": 15
    }

    result = predict_failure(sample_device)

    print(result)