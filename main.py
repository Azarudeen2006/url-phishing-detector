import pandas as pd
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def extract_features(url):

    features = []

    # 1. URL Length
    features.append(len(url))

    # 2. HTTPS Check
    features.append(1 if url.startswith("https") else 0)

    # 3. Number of Dots
    features.append(url.count("."))

    # 4. Number of Hyphens
    features.append(url.count("-"))

    # 5. Presence of @ Symbol
    features.append(1 if "@" in url else 0)

    # 6. Presence of IP Address
    ip_pattern = r'(\d{1,3}\.){3}\d{1,3}'
    features.append(1 if re.search(ip_pattern, url) else 0)

    # 7. Suspicious Keywords
    keywords = [
        "login",
        "verify",
        "secure",
        "update",
        "bank",
        "account",
        "paypal",
        "signin",
        "password",
        "confirm"
    ]

    keyword_count = 0

    for word in keywords:
        if word in url.lower():
            keyword_count += 1

    features.append(keyword_count)

    # 8. URL Shortener Detection
    shorteners = [
        "bit.ly",
        "tinyurl.com",
        "t.co",
        "goo.gl",
        "is.gd",
        "ow.ly"
    ]

    shortener_flag = 1 if any(
        shortener in url.lower()
        for shortener in shorteners
    ) else 0

    features.append(shortener_flag)

    return features


# ----------------------------
# Load Dataset
# ----------------------------

try:
    data = pd.read_csv("dataset.csv")
except FileNotFoundError:
    print("dataset.csv not found!")
    exit()

# ----------------------------
# Prepare Data
# ----------------------------

X = data["url"].apply(extract_features).tolist()
y = data["label"]

# ----------------------------
# Split Dataset
# ----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ----------------------------
# Train Model
# ----------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ----------------------------
# Evaluate Model
# ----------------------------

accuracy = model.score(X_test, y_test)

print("=" * 50)
print("URL PHISHING DETECTOR")
print("=" * 50)
print(f"Model Accuracy: {accuracy:.2f}")
print("=" * 50)

# ----------------------------
# Save Model
# ----------------------------

joblib.dump(model, "model.pkl")

# ----------------------------
# User Input
# ----------------------------

url = input("Enter URL: ").strip()

if not url:
    print("Please enter a valid URL.")
    exit()

# ----------------------------
# Prediction
# ----------------------------

features = [extract_features(url)]

prediction = model.predict(features)

probability = model.predict_proba(features)[0][1]

risk_score = probability * 100

print("\n----- RESULT -----")
print(f"Risk Score: {risk_score:.2f}%")

if prediction[0] == 1:
    print("⚠️ Phishing URL Detected")
else:
    print("✅ Safe URL")