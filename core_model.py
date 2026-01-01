import pandas as pd
import numpy as np
import json
import pickle
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.metrics import precision_recall_curve, roc_curve
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    
    def __init__(self):
        self.feature_names = []
        self.categorical_mappings = {}
        
    def create_features(self, df):
        df = df.copy()
        
        df["final_result"] = df["G3"].apply(lambda x: 1 if x >= 10 else 0)
        
        numeric_features = ["age", "studytime", "failures", "absences", "G1", "G2"]
        
        self.categorical_mappings = {
            "school": {"GP": 1, "MS": 0},
            "sex": {"F": 1, "M": 0},
            "address": {"U": 1, "R": 0},
            "famsize": {"GT3": 1, "LE3": 0},
            "Pstatus": {"T": 1, "A": 0}
        }
        
        for col, mapping in self.categorical_mappings.items():
            df[f"{col}_encoded"] = df[col].map(mapping)
        
        education_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
        df["Medu_encoded"] = df["Medu"].map(education_map)
        df["Fedu_encoded"] = df["Fedu"].map(education_map)
        
        binary_features = ["schoolsup", "famsup", "paid", "activities", 
                          "nursery", "higher", "internet", "romantic"]
        for feat in binary_features:
            df[f"{feat}_encoded"] = df[feat].map({"yes": 1, "no": 0})
        
        df["parent_edu_avg"] = (df["Medu_encoded"] + df["Fedu_encoded"]) / 2
        df["parent_edu_max"] = df[["Medu_encoded", "Fedu_encoded"]].max(axis=1)
        df["total_support"] = df["schoolsup_encoded"] + df["famsup_encoded"]
        df["study_quality"] = df["studytime"] * (5 - df["failures"])
        df["academic_trend"] = df["G2"] - df["G1"]
        df["avg_grade"] = (df["G1"] + df["G2"]) / 2
        df["grade_volatility"] = abs(df["G2"] - df["G1"])
        df["family_quality"] = df["parent_edu_avg"] + df["famrel"] + df["total_support"]
        df["social_score"] = df["goout"] + df["Dalc"] + df["Walc"]
        df["health_attendance"] = df["health"] - (df["absences"] / 10)
        
        self.feature_names = (
            numeric_features + 
            ["school_encoded", "sex_encoded", "address_encoded", "famsize_encoded", 
             "Pstatus_encoded", "Medu_encoded", "Fedu_encoded"] +
            [f"{feat}_encoded" for feat in binary_features] +
            ["parent_edu_avg", "parent_edu_max", "total_support", "study_quality", 
             "academic_trend", "avg_grade", "grade_volatility", "family_quality",
             "social_score", "health_attendance"]
        )
        
        X = df[self.feature_names]
        y = df["final_result"]
        
        return X, y, df


class ModelTrainer:
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.scaler = None
        self.feature_names = None
        self.metrics = {}
        
    def train(self, X, y, feature_names):
        self.feature_names = feature_names
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        lr = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
        rf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, 
                                     min_samples_split=5, min_samples_leaf=2)
        gb = GradientBoostingClassifier(n_estimators=150, max_depth=5, random_state=42,
                                        learning_rate=0.1)
        
        self.models = {
            "Logistic Regression": lr,
            "Random Forest": rf,
            "Gradient Boosting": gb
        }
        
        for name, model in self.models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            self.metrics[name] = {
                "accuracy": accuracy_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
            }
        
        ensemble = VotingClassifier(
            estimators=[(name, model) for name, model in self.models.items()],
            voting='soft'
        )
        ensemble.fit(X_train_scaled, y_train)
        y_pred_ensemble = ensemble.predict(X_test_scaled)
        
        self.metrics["Ensemble"] = {
            "accuracy": accuracy_score(y_test, y_pred_ensemble),
            "f1": f1_score(y_test, y_pred_ensemble),
            "roc_auc": roc_auc_score(y_test, ensemble.predict_proba(X_test_scaled)[:, 1])
        }
        
        self.models["Ensemble"] = ensemble
        
        best_name = max(self.metrics.items(), key=lambda x: x[1]["roc_auc"])[0]
        self.best_model = self.models[best_name]
        
        return {
            "train_size": len(X_train),
            "test_size": len(X_test),
            "best_model": best_name,
            "metrics": self.metrics,
            "X_test": X_test_scaled,
            "y_test": y_test
        }
    
    def predict_with_confidence(self, student_data):
        student_scaled = self.scaler.transform([student_data])
        prediction = self.best_model.predict(student_scaled)[0]
        probabilities = self.best_model.predict_proba(student_scaled)[0]
        
        confidence = max(probabilities) * 100
        
        feature_importance = None
        if hasattr(self.best_model, 'feature_importances_'):
            feature_importance = dict(zip(self.feature_names, 
                                         self.best_model.feature_importances_))
        elif hasattr(self.best_model, 'estimators_'):
            importances = []
            for estimator in self.best_model.estimators_:
                if hasattr(estimator, 'feature_importances_'):
                    importances.append(estimator.feature_importances_)
            if importances:
                avg_importance = np.mean(importances, axis=0)
                feature_importance = dict(zip(self.feature_names, avg_importance))
        
        return {
            "prediction": int(prediction),
            "result": "PASS" if prediction == 1 else "FAIL",
            "confidence": confidence,
            "probabilities": {
                "fail": float(probabilities[0]),
                "pass": float(probabilities[1])
            },
            "feature_importance": feature_importance
        }


class WhatIfSimulator:
    
    def __init__(self, model, scaler, feature_names):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
    
    def simulate(self, base_data):
        simulations = {
            "studytime": [("Increase study time +1", 1), ("Increase study time +2", 2)],
            "absences": [("Reduce absences -5", -5), ("Reduce absences -10", -10)],
            "G1": [("Improve G1 +2", 2), ("Improve G1 +3", 3)],
            "G2": [("Improve G2 +2", 2), ("Improve G2 +3", 3)],
            "failures": [("Remove 1 failure", -1)],
        }
        
        base_scaled = self.scaler.transform([base_data])
        base_prob = self.model.predict_proba(base_scaled)[0][1]
        
        results = {"baseline": float(base_prob), "scenarios": []}
        
        for feature, scenarios in simulations.items():
            if feature in self.feature_names:
                idx = self.feature_names.index(feature)
                
                for description, change in scenarios:
                    modified_data = base_data.copy()
                    modified_data[idx] = max(0, min(20, modified_data[idx] + change))
                    
                    modified_scaled = self.scaler.transform([modified_data])
                    new_prob = self.model.predict_proba(modified_scaled)[0][1]
                    
                    results["scenarios"].append({
                        "description": description,
                        "feature": feature,
                        "change": change,
                        "new_probability": float(new_prob),
                        "impact": float(new_prob - base_prob)
                    })
        
        results["scenarios"].sort(key=lambda x: abs(x["impact"]), reverse=True)
        return results


class HybridDecisionSystem:
    
    def __init__(self, feature_names):
        self.feature_names = feature_names
        self.risk_thresholds = {
            "failures": 2,
            "absences": 20,
            "G2": 8,
            "confidence": 60
        }
    
    def evaluate(self, prediction_result, student_data):
        flags = []
        risk_score = 0
        
        failures_idx = self.feature_names.index("failures")
        absences_idx = self.feature_names.index("absences")
        g2_idx = self.feature_names.index("G2")
        
        failures = student_data[failures_idx]
        absences = student_data[absences_idx]
        g2_score = student_data[g2_idx]
        confidence = prediction_result["confidence"]
        
        if failures >= self.risk_thresholds["failures"]:
            flags.append({"type": "HIGH_RISK", "message": f"Student has {int(failures)} previous failures"})
            risk_score += 30
        
        if absences > self.risk_thresholds["absences"]:
            flags.append({"type": "ATTENDANCE", "message": f"{int(absences)} absences detected"})
            risk_score += 25
        
        if g2_score < self.risk_thresholds["G2"]:
            flags.append({"type": "ACADEMIC", "message": f"Low G2 score: {g2_score:.1f}"})
            risk_score += 20
        
        if confidence < self.risk_thresholds["confidence"]:
            flags.append({"type": "LOW_CONFIDENCE", "message": f"AI confidence only {confidence:.1f}%"})
            risk_score += 15
        
        if prediction_result["result"] == "FAIL" and confidence > 80:
            flags.append({"type": "CRITICAL", "message": "High confidence FAIL prediction"})
            risk_score += 35
        
        if risk_score >= 50:
            decision = "MANUAL_REVIEW_REQUIRED"
            recommendation = "Schedule immediate intervention"
        elif risk_score >= 30:
            decision = "MONITORING_RECOMMENDED"
            recommendation = "Increase academic support"
        else:
            decision = prediction_result["result"]
            recommendation = "Continue current plan"
        
        return {
            "final_decision": decision,
            "risk_score": risk_score,
            "flags": flags,
            "recommendation": recommendation,
            "ai_prediction": prediction_result["result"],
            "requires_human_review": len(flags) > 0
        }


class FeedbackMemory:
    
    def __init__(self, storage_path="data/feedback_memory.json"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(exist_ok=True)
        self.memory = self._load()
    
    def _load(self):
        if self.storage_path.exists():
            with open(self.storage_path, 'r') as f:
                return json.load(f)
        return {"predictions": [], "corrections": [], "statistics": {}}
    
    def _save(self):
        with open(self.storage_path, 'w') as f:
            json.dump(self.memory, f, indent=2)
    
    def add_prediction(self, student_id, prediction_data, student_features):
        entry = {
            "id": student_id,
            "timestamp": datetime.now().isoformat(),
            "prediction": prediction_data,
            "features": student_features,
            "actual_result": None,
            "corrected": False
        }
        self.memory["predictions"].append(entry)
        self._save()
        return entry
    
    def add_correction(self, student_id, actual_result):
        for pred in self.memory["predictions"]:
            if pred["id"] == student_id and not pred["corrected"]:
                pred["actual_result"] = actual_result
                pred["corrected"] = True
                
                correction = {
                    "id": student_id,
                    "timestamp": datetime.now().isoformat(),
                    "predicted": pred["prediction"]["result"],
                    "actual": actual_result,
                    "was_correct": pred["prediction"]["result"] == actual_result
                }
                self.memory["corrections"].append(correction)
                self._update_statistics()
                self._save()
                return correction
        return None
    
    def _update_statistics(self):
        total = len(self.memory["corrections"])
        if total == 0:
            return
        
        correct = sum(1 for c in self.memory["corrections"] if c["was_correct"])
        self.memory["statistics"] = {
            "total_corrections": total,
            "correct_predictions": correct,
            "accuracy": correct / total,
            "last_updated": datetime.now().isoformat()
        }
    
    def get_statistics(self):
        return self.memory["statistics"]
    
    def get_prediction_history(self, limit=50):
        return self.memory["predictions"][-limit:]


class ModelPersistence:
    
    @staticmethod
    def save(model_trainer, feature_engineer, filepath="models/student_model.pkl"):
        Path(filepath).parent.mkdir(exist_ok=True)
        
        package = {
            "model": model_trainer.best_model,
            "scaler": model_trainer.scaler,
            "feature_names": model_trainer.feature_names,
            "metrics": model_trainer.metrics,
            "categorical_mappings": feature_engineer.categorical_mappings,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(package, f)
        
        return filepath
    
    @staticmethod
    def load(filepath="models/student_model.pkl"):
        with open(filepath, 'rb') as f:
            package = pickle.load(f)
        
        trainer = ModelTrainer()
        trainer.best_model = package["model"]
        trainer.scaler = package["scaler"]
        trainer.feature_names = package["feature_names"]
        trainer.metrics = package["metrics"]
        
        engineer = FeatureEngineer()
        engineer.feature_names = package["feature_names"]
        engineer.categorical_mappings = package["categorical_mappings"]
        
        return trainer, engineer, package["timestamp"]