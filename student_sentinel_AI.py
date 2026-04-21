import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, r2_score

# ==========================================
# 1. DATA SIMULATION (The University Database)
# ==========================================
np.random.seed(42)
n = 1000

# Define our professional feature names
feature_names = ['wifi_library_hours', 'lms_clicks', 'commute_km', 'interest_gap']

# Simulating raw behavior
data = {
    'wifi_library_hours': np.random.normal(40, 15, n),  # Gaussian
    'lms_clicks': np.random.poisson(50, n),  # Poisson
    'commute_km': np.random.uniform(2, 60, n),  # Uniform
    'interest_gap': np.random.normal(0, 1, n)  # Gaussian (Field Mismatch)
}

df = pd.DataFrame(data)

# --- The Mathematical Truth (Target Generation) ---
# We define the "Physics" of our world: GPA is a linear combination of inputs
df['predicted_gpa'] = (
        (df['wifi_library_hours'] * 0.02) +
        (df['lms_clicks'] * 0.01) -
        (df['commute_km'] * 0.01) -
        (df['interest_gap'] * 0.5) +
        np.random.normal(2.5, 0.3, n)  # The Noise (Stochastic Factor)
)
df['predicted_gpa'] = df['predicted_gpa'].clip(0, 4.0)

# Binary Label for Classification (Risk = 1 if GPA < 2.2)
df['risk_status'] = (df['predicted_gpa'] < 2.2).astype(int)

# ==========================================
# 2. THE AI PIPELINE (Training)
# ==========================================
X = df[feature_names]
y_class = df['risk_status']
y_reg = df['predicted_gpa']

# The Golden Rule: Train-Test Split
X_train, X_test, y_train_c, y_test_c, y_train_r, y_test_r = train_test_split(
    X, y_class, y_reg, test_size=0.2, random_state=42
)

# Scaling (Standardization/Z-Score)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and Train Models
classifier = LogisticRegression()
classifier.fit(X_train_scaled, y_train_c)

regressor = LinearRegression()
regressor.fit(X_train_scaled, y_train_r)


# ==========================================
# 3. INTERVENTION & REPORTING TOOLS
# ==========================================

def assess_single_student(wifi, clicks, km, gap):
    """Predicts status for an individual student."""
    input_df = pd.DataFrame([[wifi, clicks, km, gap]], columns=feature_names)
    scaled_input = scaler.transform(input_df)

    prob = classifier.predict_proba(scaled_input)[0][1]
    gpa_pred = regressor.predict(scaled_input)[0]

    print("\n" + "=" * 30)
    print("INDIVIDUAL ASSESSMENT REPORT")
    print("=" * 30)
    print(f"Predicted GPA:    {gpa_pred:.2f}")
    print(f"Risk Probability: {prob:.2%}")
    print(f"Recommendation:   {'IMMEDIATE COUNSELING' if prob > 0.6 else 'STABLE'}")


def export_at_risk_report(test_scaled, test_original):
    """Generates a CSV file for the University Dean."""
    probs = classifier.predict_proba(test_scaled)[:, 1]

    # Create the report
    report = test_original.copy()
    report['Risk_Probability'] = probs

    # Filter for those above 50% risk and sort by urgency
    at_risk_list = report[report['Risk_Probability'] > 0.5].sort_values(by='Risk_Probability', ascending=False)

    # Save file
    at_risk_list.to_csv("Priority_Counseling_List.csv", index=False)

    print("\n" + "=" * 30)
    print("BULK SYSTEM STATUS")
    print("=" * 30)
    print(f"Process complete. CSV generated.")
    print(f"High-Risk students identified: {len(at_risk_list)}")


# ==========================================
# 4. EXECUTION
# ==========================================

if __name__ == "__main__":
    # 1. Individual Inference (Manual Check)
    assess_single_student(wifi=10, clicks=5, km=55, gap=2.5)

    # 2. System-wide Reporting (The Dean's File)
    export_at_risk_report(X_test_scaled, X_test)

    # 3. Validation Metrics (The Math Check)
    r2 = r2_score(y_test_r, regressor.predict(X_test_scaled))
    print(f"Model Reliability (R-Squared): {r2:.2%}")