# ============================================
# STEP 4: LOAD THE DATA (FIXED)
# ============================================

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load CSV with correct separator
data_path = "data/student-mat.csv"
df = pd.read_csv(data_path, sep=";")

# Check columns
print("Columns in dataset:")
print(df.columns)

print("\nFirst 5 rows:")
print(df.head())


# ============================================
# STEP 5: CLEAN THE DATA
# ============================================

# 1️⃣ Remove empty rows
df = df.dropna()
print("\nShape after removing empty rows:", df.shape)


# 2️⃣ Convert G3 (final marks) → PASS / FAIL
# PASS = 1, FAIL = 0
df["final_result"] = df["G3"].apply(lambda x: 1 if x >= 10 else 0)

print("\nPASS / FAIL count:")
print(df["final_result"].value_counts())


# 3️⃣ Select useful columns
features = ["age", "studytime", "failures", "absences", "G1", "G2"]
X = df[features]
y = df["final_result"]


# 4️⃣ Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_scaled = pd.DataFrame(X_scaled, columns=features)

print("\nNormalized feature preview:")
print(X_scaled.head())

print("\n✅ DATA LOADING & CLEANING SUCCESSFUL!")
# ============================================
# COMPLETE STUDENT PERFORMANCE PREDICTION SYSTEM
# With Explainability, Confidence, What-if & Human-in-the-loop
# ============================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ============================================
# STEP 1: LOAD & PREPARE DATA
# ============================================

def load_and_prepare_data(data_path="data/student-mat.csv"):
    """Load and prepare the student performance dataset"""
    
    # Load CSV
    df = pd.read_csv(data_path, sep=";")
    print("✅ Dataset loaded successfully!")
    print(f"Total students: {len(df)}")
    
    # Remove empty rows
    df = df.dropna()
    
    # Create target variable (PASS = 1 if G3 >= 10, else FAIL = 0)
    df["final_result"] = df["G3"].apply(lambda x: 1 if x >= 10 else 0)
    
    print(f"\n📊 Class Distribution:")
    print(f"  PASS: {sum(df['final_result'] == 1)} students")
    print(f"  FAIL: {sum(df['final_result'] == 0)} students")
    
    return df


# ============================================
# STEP 2: FEATURE ENGINEERING
# ============================================

def engineer_features(df):
    """Create enhanced features for better predictions"""
    
    # Numeric features
    numeric_features = ["age", "studytime", "failures", "absences", "G1", "G2"]
    
    # Convert categorical to numeric
    df["school_encoded"] = df["school"].map({"GP": 1, "MS": 0})
    df["sex_encoded"] = df["sex"].map({"F": 1, "M": 0})
    df["address_encoded"] = df["address"].map({"U": 1, "R": 0})
    df["famsize_encoded"] = df["famsize"].map({"GT3": 1, "LE3": 0})
    df["Pstatus_encoded"] = df["Pstatus"].map({"T": 1, "A": 0})
    
    # Education level encoding (higher = more education)
    education_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
    df["Medu_encoded"] = df["Medu"].map(education_map)
    df["Fedu_encoded"] = df["Fedu"].map(education_map)
    
    # Binary yes/no features
    binary_features = ["schoolsup", "famsup", "paid", "activities", 
                      "nursery", "higher", "internet", "romantic"]
    for feat in binary_features:
        df[f"{feat}_encoded"] = df[feat].map({"yes": 1, "no": 0})
    
    # Create composite features
    df["parent_edu_avg"] = (df["Medu_encoded"] + df["Fedu_encoded"]) / 2
    df["total_support"] = df["schoolsup_encoded"] + df["famsup_encoded"]
    df["study_quality"] = df["studytime"] * (5 - df["failures"])  # More study, fewer failures = better
    df["academic_trend"] = df["G2"] - df["G1"]  # Improvement trend
    
    # All feature columns
    all_features = (numeric_features + 
                   ["school_encoded", "sex_encoded", "address_encoded", 
                    "famsize_encoded", "Pstatus_encoded", "Medu_encoded", 
                    "Fedu_encoded", "schoolsup_encoded", "famsup_encoded", 
                    "paid_encoded", "activities_encoded", "nursery_encoded",
                    "higher_encoded", "internet_encoded", "romantic_encoded",
                    "parent_edu_avg", "total_support", "study_quality", "academic_trend"])
    
    X = df[all_features]
    y = df["final_result"]
    
    print(f"\n🔧 Feature Engineering Complete!")
    print(f"Total features: {len(all_features)}")
    
    return X, y, all_features, df


# ============================================
# STEP 3: TRAIN MODELS
# ============================================

def train_models(X, y):
    """Train multiple models and return the best one"""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Normalize features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n🤖 Training Models...")
    
    # Model 1: Logistic Regression
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    lr_accuracy = accuracy_score(y_test, lr_pred)
    
    # Model 2: Random Forest (Better for feature importance)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    
    print(f"\n📈 Model Performance:")
    print(f"  Logistic Regression Accuracy: {lr_accuracy:.2%}")
    print(f"  Random Forest Accuracy: {rf_accuracy:.2%}")
    
    # Choose best model
    best_model = rf_model if rf_accuracy > lr_accuracy else lr_model
    best_name = "Random Forest" if rf_accuracy > lr_accuracy else "Logistic Regression"
    best_accuracy = max(lr_accuracy, rf_accuracy)
    
    print(f"\n✅ Best Model: {best_name} ({best_accuracy:.2%})")
    
    # Detailed metrics
    print("\n📊 Classification Report:")
    print(classification_report(y_test, rf_pred, 
                                target_names=["FAIL", "PASS"]))
    
    return best_model, scaler, X_test_scaled, y_test, rf_model


# ============================================
# STEP 4: CONFIDENCE & EXPLANATION
# ============================================

def predict_with_confidence(model, scaler, features, feature_names, student_data):
    """Make prediction with confidence score and explanation"""
    
    # Scale the input
    student_scaled = scaler.transform([student_data])
    
    # Get prediction and probability
    prediction = model.predict(student_scaled)[0]
    probabilities = model.predict_proba(student_scaled)[0]
    
    confidence = max(probabilities) * 100
    result = "PASS" if prediction == 1 else "FAIL"
    
    print("\n" + "="*50)
    print("🎯 PREDICTION RESULT")
    print("="*50)
    print(f"Prediction: {result}")
    print(f"Confidence: {confidence:.1f}%")
    print(f"  - Probability of FAIL: {probabilities[0]:.1%}")
    print(f"  - Probability of PASS: {probabilities[1]:.1%}")
    
    # Feature importance (if Random Forest)
    if hasattr(model, 'feature_importances_'):
        print("\n📊 TOP FACTORS INFLUENCING THIS PREDICTION:")
        importances = model.feature_importances_
        top_indices = np.argsort(importances)[-5:][::-1]
        
        for i, idx in enumerate(top_indices, 1):
            feature_name = feature_names[idx]
            feature_value = student_data[idx]
            importance = importances[idx] * 100
            print(f"  {i}. {feature_name}: {feature_value:.2f} (importance: {importance:.1f}%)")
    
    return prediction, confidence, probabilities


# ============================================
# STEP 5: WHAT-IF SIMULATION
# ============================================

def what_if_simulation(model, scaler, base_data, feature_names):
    """Simulate changes to key features"""
    
    print("\n" + "="*50)
    print("🔮 WHAT-IF SIMULATION")
    print("="*50)
    
    # Key features to simulate
    simulations = {
        "Increase study time by 1 hour": ("studytime", 1),
        "Reduce absences by 5": ("absences", -5),
        "Improve G1 score by 2 points": ("G1", 2),
        "Improve G2 score by 2 points": ("G2", 2),
        "Remove 1 failure": ("failures", -1),
    }
    
    # Get baseline prediction
    base_scaled = scaler.transform([base_data])
    base_prob = model.predict_proba(base_scaled)[0][1]
    
    print(f"\nBaseline PASS probability: {base_prob:.1%}")
    print("\nImpact of changes:")
    
    for description, (feature, change) in simulations.items():
        # Find feature index
        if feature in feature_names:
            idx = feature_names.index(feature)
            
            # Create modified data
            modified_data = base_data.copy()
            modified_data[idx] = max(0, modified_data[idx] + change)
            
            # Get new prediction
            modified_scaled = scaler.transform([modified_data])
            new_prob = model.predict_proba(modified_scaled)[0][1]
            
            change_pct = (new_prob - base_prob) * 100
            arrow = "📈" if change_pct > 0 else "📉"
            
            print(f"  {arrow} {description}")
            print(f"     New probability: {new_prob:.1%} ({change_pct:+.1f} percentage points)")


# ============================================
# STEP 6: HUMAN-IN-THE-LOOP DECISION
# ============================================

def hybrid_decision_system(prediction, confidence, student_data, feature_names):
    """Combine AI prediction with human oversight rules"""
    
    print("\n" + "="*50)
    print("🤝 HYBRID DECISION SYSTEM (AI + HUMAN RULES)")
    print("="*50)
    
    # Extract key features
    failures_idx = feature_names.index("failures")
    absences_idx = feature_names.index("absences")
    g2_idx = feature_names.index("G2")
    
    failures = student_data[failures_idx]
    absences = student_data[absences_idx]
    g2_score = student_data[g2_idx]
    
    # Human-defined rules
    flags = []
    
    if failures >= 2:
        flags.append("⚠️  HIGH RISK: Student has 2+ previous failures")
    
    if absences > 20:
        flags.append("⚠️  ATTENDANCE CONCERN: 20+ absences")
    
    if g2_score < 8:
        flags.append("⚠️  LOW MID-TERM SCORE: G2 < 8")
    
    if confidence < 60:
        flags.append("⚠️  LOW CONFIDENCE: AI is less than 60% confident")
    
    # Final decision
    if flags:
        print("\n🚨 MANUAL REVIEW RECOMMENDED:")
        for flag in flags:
            print(f"  {flag}")
        print("\n💡 Recommendation: Schedule intervention or counseling")
        final_decision = "REVIEW NEEDED"
    else:
        result = "PASS" if prediction == 1 else "FAIL"
        print(f"\n✅ AI Decision Accepted: {result}")
        print("   No critical risk factors detected")
        final_decision = result
    
    return final_decision, flags


# ============================================
# STEP 7: MAIN EXECUTION
# ============================================

def main():
    """Run the complete system"""
    
    print("🎓 STUDENT PERFORMANCE PREDICTION SYSTEM")
    print("="*50)
    
    # Load data
    df = load_and_prepare_data()
    
    # Engineer features
    X, y, feature_names, df_full = engineer_features(df)
    
    # Train models
    best_model, scaler, X_test, y_test, rf_model = train_models(X, y)
    
    # ============================================
    # EXAMPLE: PREDICT FOR A NEW STUDENT
    # ============================================
    
    print("\n" + "="*50)
    print("🧪 TESTING WITH SAMPLE STUDENT")
    print("="*50)
    
    # Example student data (you can modify this)
    sample_student = {
        "age": 17,
        "studytime": 2,      # 1-4 scale
        "failures": 0,
        "absences": 6,
        "G1": 11,           # First period grade
        "G2": 12,           # Second period grade
        "school": "GP",
        "sex": "F",
        "address": "U",
        "famsize": "GT3",
        "Pstatus": "T",
        "Medu": 3,
        "Fedu": 3,
        "schoolsup": "no",
        "famsup": "yes",
        "paid": "no",
        "activities": "yes",
        "nursery": "yes",
        "higher": "yes",
        "internet": "yes",
        "romantic": "no"
    }
    
    # Create feature vector for sample student
    sample_features = []
    
    # Numeric features
    sample_features.extend([
        sample_student["age"],
        sample_student["studytime"],
        sample_student["failures"],
        sample_student["absences"],
        sample_student["G1"],
        sample_student["G2"]
    ])
    
    # Encoded categorical features
    sample_features.extend([
        1 if sample_student["school"] == "GP" else 0,
        1 if sample_student["sex"] == "F" else 0,
        1 if sample_student["address"] == "U" else 0,
        1 if sample_student["famsize"] == "GT3" else 0,
        1 if sample_student["Pstatus"] == "T" else 0,
        sample_student["Medu"],
        sample_student["Fedu"],
        1 if sample_student["schoolsup"] == "yes" else 0,
        1 if sample_student["famsup"] == "yes" else 0,
        1 if sample_student["paid"] == "yes" else 0,
        1 if sample_student["activities"] == "yes" else 0,
        1 if sample_student["nursery"] == "yes" else 0,
        1 if sample_student["higher"] == "yes" else 0,
        1 if sample_student["internet"] == "yes" else 0,
        1 if sample_student["romantic"] == "yes" else 0,
    ])
    
    # Composite features
    parent_edu_avg = (sample_student["Medu"] + sample_student["Fedu"]) / 2
    total_support = (1 if sample_student["schoolsup"] == "yes" else 0) + (1 if sample_student["famsup"] == "yes" else 0)
    study_quality = sample_student["studytime"] * (5 - sample_student["failures"])
    academic_trend = sample_student["G2"] - sample_student["G1"]
    
    sample_features.extend([parent_edu_avg, total_support, study_quality, academic_trend])
    
    # Make prediction
    prediction, confidence, probabilities = predict_with_confidence(
        rf_model, scaler, sample_features, feature_names, sample_features
    )
    
    # What-if simulation
    what_if_simulation(rf_model, scaler, sample_features, feature_names)
    
    # Hybrid decision
    final_decision, flags = hybrid_decision_system(
        prediction, confidence, sample_features, feature_names
    )
    
    print("\n" + "="*50)
    print("🎯 FINAL DECISION:", final_decision)
    print("="*50)
    
    print("\n✅ System execution complete!")


# ============================================
# RUN THE SYSTEM
# ============================================

if __name__ == "__main__":
    main()
