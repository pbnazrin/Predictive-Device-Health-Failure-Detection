import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# 1. Configuration
# ============================================================

np.random.seed(42)

N_DEVICES = 12000


# ============================================================
# 2. Helper function
# ============================================================

def sigmoid(x):
    """Convert a risk score into a probability between 0 and 1."""
    return 1 / (1 + np.exp(-x))


# ============================================================
# 3. Generate basic device information
# ============================================================

df = pd.DataFrame({
    "Device_ID": [
        f"DEV-{i:05d}"
        for i in range(1, N_DEVICES + 1)
    ],

    "OS": np.random.choice(
        ["Android", "Windows", "iOS"],
        size=N_DEVICES,
        p=[0.56, 0.29, 0.15]
    ),

    "OS_Version": np.random.choice(
        [10, 11, 12, 13, 14, 15],
        size=N_DEVICES,
        p=[0.04, 0.16, 0.25, 0.25, 0.20, 0.10]
    ),

    "Manufacturer": np.random.choice(
        [
            "Samsung",
            "Lenovo",
            "Dell",
            "Apple",
            "Zebra",
            "Honeywell"
        ],
        size=N_DEVICES,
        p=[0.22, 0.18, 0.18, 0.14, 0.16, 0.12]
    ),

    "Model": np.random.choice(
        [
            "Model_A",
            "Model_B",
            "Model_C",
            "Model_D",
            "Model_E",
            "Model_F"
        ],
        size=N_DEVICES
    ),

    "Network_Type": np.random.choice(
        ["WiFi", "4G", "5G", "Ethernet"],
        size=N_DEVICES,
        p=[0.42, 0.28, 0.18, 0.12]
    )
})


# ============================================================
# 4. Generate device telemetry
# ============================================================

df["Battery_Health"] = np.clip(
    np.random.normal(78, 12, N_DEVICES),
    20,
    100
)

df["Battery_Temperature"] = np.clip(
    np.random.normal(35, 4.5, N_DEVICES),
    20,
    55
)

df["CPU_Usage"] = np.clip(
    np.random.normal(55, 18, N_DEVICES),
    5,
    100
)

df["Memory_Usage"] = np.clip(
    np.random.normal(60, 16, N_DEVICES),
    10,
    100
)

df["Storage_Usage"] = np.clip(
    np.random.normal(65, 18, N_DEVICES),
    10,
    100
)

df["Network_Signal"] = np.clip(
    np.random.normal(70, 18, N_DEVICES),
    5,
    100
)

df["App_Crash_Count"] = np.random.poisson(
    2.2,
    N_DEVICES
)

df["Reboot_Count"] = np.random.poisson(
    2.0,
    N_DEVICES
)

df["Error_Count"] = np.random.poisson(
    3.0,
    N_DEVICES
)

df["Data_Usage"] = np.random.lognormal(
    mean=np.log(1800),
    sigma=0.55,
    size=N_DEVICES
)

df["Days_Since_Last_Update"] = np.random.randint(
    1,
    181,
    N_DEVICES
)

df["Last_Checkin_Hours"] = np.random.exponential(
    4.5,
    N_DEVICES
)


# ============================================================
# 5. Create meaningful abnormal device conditions
# ============================================================

# Some devices have severe overheating
hot_indices = np.random.choice(
    N_DEVICES,
    size=45,
    replace=False
)

df.loc[
    hot_indices,
    "Battery_Temperature"
] = np.random.uniform(
    60,
    82,
    size=len(hot_indices)
)


# Some devices have unusually high application crashes
crash_indices = np.random.choice(
    N_DEVICES,
    size=45,
    replace=False
)

df.loc[
    crash_indices,
    "App_Crash_Count"
] = np.random.randint(
    20,
    55,
    size=len(crash_indices)
)


# Some devices have poor battery health
battery_indices = np.random.choice(
    N_DEVICES,
    size=150,
    replace=False
)

df.loc[
    battery_indices,
    "Battery_Health"
] = np.random.uniform(
    20,
    45,
    size=len(battery_indices)
)


# ============================================================
# 6. Create predictive failure risk
# ============================================================
#
# IMPORTANT:
#
# We intentionally make the target depend strongly enough
# on device-health features so that ML models can learn
# meaningful patterns.
#
# Higher risk:
#   - Low battery health
#   - High temperature
#   - High CPU
#   - High memory
#   - High app crashes
#   - High reboot count
#   - High error count
#   - Poor network signal
#   - Old software update
#   - Long time since check-in
#

risk_score = (

    # Base risk
    -3.0

    # Battery degradation
    + 0.070 * (80 - df["Battery_Health"])

    # Overheating
    + 0.100 * (df["Battery_Temperature"] - 35)

    # CPU pressure
    + 0.045 * (df["CPU_Usage"] - 55)

    # Memory pressure
    + 0.055 * (df["Memory_Usage"] - 60)

    # Storage pressure
    + 0.020 * (df["Storage_Usage"] - 65)

    # Poor network signal
    + 0.035 * (70 - df["Network_Signal"])

    # Application instability
    + 0.20 * df["App_Crash_Count"]

    # Frequent rebooting
    + 0.13 * df["Reboot_Count"]

    # Device errors
    + 0.18 * df["Error_Count"]

    # Old software
    + 0.018 * (df["Days_Since_Last_Update"] - 90)

    # Device not checking in
    + 0.10 * (df["Last_Checkin_Hours"] - 5)

    # --------------------------------------------------------
    # Interaction effects
    # --------------------------------------------------------

    # High CPU + high memory is more dangerous together
    + 0.0015
      * np.maximum(df["CPU_Usage"] - 70, 0)
      * np.maximum(df["Memory_Usage"] - 70, 0)

    # High temperature becomes much more dangerous above 50°C
    + 0.040
      * np.maximum(df["Battery_Temperature"] - 50, 0) ** 1.4

    # Very low battery health increases risk sharply
    + 0.080
      * np.maximum(50 - df["Battery_Health"], 0)

    # Very poor network + long check-in gap
    + 0.025
      * np.maximum(40 - df["Network_Signal"], 0)
      * np.maximum(df["Last_Checkin_Hours"] - 10, 0)
)


# ============================================================
# 7. Convert risk score to probability
# ============================================================

failure_probability = sigmoid(risk_score)


# ============================================================
# 8. Generate target
# ============================================================

df["Failure_Within_7_Days"] = np.random.binomial(
    n=1,
    p=failure_probability
)


# ============================================================
# 9. Add missing values
# ============================================================

missing_columns = [
    "Battery_Health",
    "Battery_Temperature",
    "Network_Signal",
    "OS_Version"
]

for column in missing_columns:

    missing_indices = np.random.choice(
        N_DEVICES,
        size=int(N_DEVICES * 0.02),
        replace=False
    )

    df.loc[
        missing_indices,
        column
    ] = np.nan


# ============================================================
# 10. Add duplicate rows
# ============================================================

duplicate_rows = df.sample(
    20,
    random_state=123
)

df = pd.concat(
    [df, duplicate_rows],
    ignore_index=True
)


# ============================================================
# 11. Save dataset
# ============================================================

output_path = Path(
    "../data/raw/device_telemetry.csv"
)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    output_path,
    index=False
)


# ============================================================
# 12. Print dataset information
# ============================================================

print("=" * 60)
print("Device Health Dataset Generated")
print("=" * 60)

print(f"Shape: {df.shape}")

print(
    f"Failure rate: "
    f"{df['Failure_Within_7_Days'].mean() * 100:.2f}%"
)

print(
    f"Duplicate rows: "
    f"{df.duplicated().sum()}"
)

print("\nMissing values:")
print(
    df.isnull().sum()[
        df.isnull().sum() > 0
    ]
)

print("\nTarget distribution:")
print(
    df["Failure_Within_7_Days"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nDataset saved to:")
print(output_path)

print("=" * 60)