import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

from core_model import (FeatureEngineer, ModelTrainer, WhatIfSimulator, 
                        HybridDecisionSystem, FeedbackMemory, ModelPersistence)
from utils import (DataExporter, EmailAlertSystem, VisualizationEngine, 
                   BatchProcessor, CompetitionFormatter)


st.set_page_config(
    page_title="Student Performance Prediction System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_model():
    try:
        trainer, engineer, timestamp = ModelPersistence.load()
        return trainer, engineer, timestamp, None
    except:
        return None, None, None, "No trained model found"


@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/student-mat.csv", sep=";")
        return df, None
    except Exception as e:
        return None, str(e)


def train_new_model():
    with st.spinner("Training models..."):
        df, error = load_data()
        if error:
            st.error(f"Error loading data: {error}")
            return None, None
        
        engineer = FeatureEngineer()
        X, y, df_processed = engineer.create_features(df)
        
        trainer = ModelTrainer()
        results = trainer.train(X, y, engineer.feature_names)
        
        ModelPersistence.save(trainer, engineer)
        
        return trainer, engineer, results


def main():
    st.title("🎓 Student Performance Prediction System")
    st.markdown("### Advanced ML System with Explainability & Human-in-the-Loop")
    
    with st.sidebar:
        st.header("⚙️ System Control")
        
        page = st.radio("Navigation", [
            "🏠 Home",
            "🤖 Single Prediction",
            "📊 Batch Analysis",
            "📈 Model Performance",
            "🔄 Feedback & Corrections",
            "⚙️ Settings & Export"
        ])
        
        st.divider()
        
        if st.button("🔄 Train New Model"):
            trainer, engineer, results = train_new_model()
            if trainer:
                st.success("✅ Model trained successfully!")
                st.rerun()
    
    trainer, engineer, model_timestamp, error = load_model()
    
    if error:
        st.warning("⚠️ No trained model found. Please train a model first.")
        if st.button("🚀 Train Initial Model"):
            trainer, engineer, results = train_new_model()
            if trainer:
                st.success("✅ Model trained successfully!")
                st.rerun()
        return
    
    if page == "🏠 Home":
        show_home(trainer, engineer, model_timestamp)
    
    elif page == "🤖 Single Prediction":
        show_single_prediction(trainer, engineer)
    
    elif page == "📊 Batch Analysis":
        show_batch_analysis(trainer, engineer)
    
    elif page == "📈 Model Performance":
        show_model_performance(trainer, engineer)
    
    elif page == "🔄 Feedback & Corrections":
        show_feedback_system()
    
    elif page == "⚙️ Settings & Export":
        show_settings_export(trainer, engineer)


def show_home(trainer, engineer, model_timestamp):
    st.header("System Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Model Loaded", "✅ Active")
    with col2:
        st.metric("Features", len(trainer.feature_names))
    with col3:
        best_acc = max(m["accuracy"] for m in trainer.metrics.values())
        st.metric("Best Accuracy", f"{best_acc:.2%}")
    with col4:
        st.metric("Training Date", model_timestamp.split('T')[0] if model_timestamp else "N/A")
    
    st.divider()
    
    st.subheader("📊 Model Comparison")
    metrics_df = pd.DataFrame(trainer.metrics).T
    metrics_df = metrics_df.round(4)
    st.dataframe(metrics_df, use_container_width=True)
    
    st.divider()
    
    st.subheader("🎯 System Capabilities")
    
    cols = st.columns(3)
    
    with cols[0]:
        st.markdown("**🔍 Explainable AI**")
        st.write("• Confidence scores")
        st.write("• Feature importance")
        st.write("• Transparent decisions")
    
    with cols[1]:
        st.markdown("**🔮 What-If Analysis**")
        st.write("• Scenario simulation")
        st.write("• Impact prediction")
        st.write("• Optimization insights")
    
    with cols[2]:
        st.markdown("**🤝 Human-in-Loop**")
        st.write("• Risk flagging")
        st.write("• Manual review triggers")
        st.write("• Feedback learning")


def show_single_prediction(trainer, engineer):
    st.header("🤖 Single Student Prediction")
    
    with st.expander("ℹ️ How to use", expanded=False):
        st.write("Enter student information below. The system will predict pass/fail and provide detailed analysis.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Student Information")
        
        age = st.slider("Age", 15, 22, 17)
        studytime = st.select_slider("Study Time (hours/week)", [1, 2, 3, 4], 2)
        failures = st.number_input("Previous Failures", 0, 4, 0)
        absences = st.number_input("Absences", 0, 93, 5)
        g1 = st.slider("G1 (First Period Grade)", 0, 20, 11)
        g2 = st.slider("G2 (Second Period Grade)", 0, 20, 12)
        
        school = st.selectbox("School", ["GP", "MS"])
        sex = st.selectbox("Sex", ["F", "M"])
        address = st.selectbox("Address Type", ["U", "R"])
        famsize = st.selectbox("Family Size", ["GT3", "LE3"])
    
    with col2:
        st.subheader("👨‍👩‍👧 Family & Support")
        
        pstatus = st.selectbox("Parent Status", ["T", "A"])
        medu = st.select_slider("Mother's Education", [0, 1, 2, 3, 4], 3)
        fedu = st.select_slider("Father's Education", [0, 1, 2, 3, 4], 3)
        
        schoolsup = st.checkbox("Educational Support")
        famsup = st.checkbox("Family Support")
        paid = st.checkbox("Extra Paid Classes")
        activities = st.checkbox("Extra-curricular Activities")
        nursery = st.checkbox("Attended Nursery")
        higher = st.checkbox("Wants Higher Education", value=True)
        internet = st.checkbox("Internet at Home", value=True)
        romantic = st.checkbox("In Relationship")
    
    if st.button("🎯 Predict Performance", type="primary"):
        
        student_data = {
            "age": age, "studytime": studytime, "failures": failures,
            "absences": absences, "G1": g1, "G2": g2,
            "school": school, "sex": sex, "address": address,
            "famsize": famsize, "Pstatus": pstatus,
            "Medu": medu, "Fedu": fedu,
            "schoolsup": "yes" if schoolsup else "no",
            "famsup": "yes" if famsup else "no",
            "paid": "yes" if paid else "no",
            "activities": "yes" if activities else "no",
            "nursery": "yes" if nursery else "no",
            "higher": "yes" if higher else "no",
            "internet": "yes" if internet else "no",
            "romantic": "yes" if romantic else "no"
        }
        
        features = create_feature_vector(student_data)
        
        prediction = trainer.predict_with_confidence(features)
        
        st.divider()
        st.subheader("🎯 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            result_color = "green" if prediction["result"] == "PASS" else "red"
            st.markdown(f"### <span style='color:{result_color}'>{prediction['result']}</span>", 
                       unsafe_allow_html=True)
            st.caption("Predicted Outcome")
        
        with col2:
            st.metric("Confidence", f"{prediction['confidence']:.1f}%")
        
        with col3:
            st.metric("Pass Probability", f"{prediction['probabilities']['pass']:.1%}")
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Probability Distribution")
            prob_data = pd.DataFrame({
                'Outcome': ['FAIL', 'PASS'],
                'Probability': [prediction['probabilities']['fail'], 
                               prediction['probabilities']['pass']]
            })
            st.bar_chart(prob_data.set_index('Outcome'))
        
        with col2:
            if prediction.get("feature_importance"):
                st.subheader("🔍 Top Influential Factors")
                top_features = sorted(prediction["feature_importance"].items(), 
                                    key=lambda x: x[1], reverse=True)[:5]
                for feat, imp in top_features:
                    st.write(f"**{feat}**: {imp:.3f}")
        
        st.divider()
        
        st.subheader("🔮 What-If Analysis")
        simulator = WhatIfSimulator(trainer.best_model, trainer.scaler, trainer.feature_names)
        what_if_results = simulator.simulate(features)
        
        st.write(f"**Baseline Pass Probability**: {what_if_results['baseline']:.1%}")
        
        scenarios_df = pd.DataFrame(what_if_results['scenarios'][:6])
        scenarios_df['impact_pct'] = scenarios_df['impact'] * 100
        scenarios_df = scenarios_df[['description', 'new_probability', 'impact_pct']]
        scenarios_df.columns = ['Scenario', 'New Probability', 'Impact (%)']
        st.dataframe(scenarios_df, use_container_width=True)
        
        st.divider()
        
        st.subheader("🤝 Hybrid Decision System")
        decision_system = HybridDecisionSystem(trainer.feature_names)
        hybrid_result = decision_system.evaluate(prediction, features)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Final Decision", hybrid_result['final_decision'])
            st.metric("Risk Score", f"{hybrid_result['risk_score']}/100")
        
        with col2:
            st.write(f"**Recommendation**: {hybrid_result['recommendation']}")
            st.write(f"**Requires Review**: {'Yes' if hybrid_result['requires_human_review'] else 'No'}")
        
        if hybrid_result['flags']:
            st.warning("⚠️ **Risk Flags Detected**:")
            for flag in hybrid_result['flags']:
                st.write(f"• [{flag['type']}] {flag['message']}")
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Save Prediction"):
                memory = FeedbackMemory()
                entry = memory.add_prediction("STU_MANUAL", prediction, features)
                st.success("✅ Prediction saved to memory!")
        
        with col2:
            if st.button("📧 Send Alert Email"):
                st.info("Configure email in Settings page first")


def show_batch_analysis(trainer, engineer):
    st.header("📊 Batch Analysis")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, sep=";")
            st.success(f"✅ Loaded {len(df)} students")
            
            st.subheader("Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            if st.button("🚀 Run Batch Prediction", type="primary"):
                with st.spinner("Processing..."):
                    X, y, df_processed = engineer.create_features(df)
                    
                    results = []
                    progress_bar = st.progress(0)
                    
                    for idx, row in X.iterrows():
                        student_data = row.values.tolist()
                        prediction = trainer.predict_with_confidence(student_data)
                        
                        results.append({
                            "student_id": f"STU_{idx:04d}",
                            "result": prediction["result"],
                            "confidence": prediction["confidence"],
                            "pass_prob": prediction["probabilities"]["pass"],
                            "fail_prob": prediction["probabilities"]["fail"]
                        })
                        
                        progress_bar.progress((idx + 1) / len(X))
                    
                    results_df = pd.DataFrame(results)
                    
                    st.success(f"✅ Processed {len(results)} students")
                    
                    st.divider()
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Students", len(results))
                    with col2:
                        pass_count = sum(1 for r in results if r['result'] == 'PASS')
                        st.metric("Predicted PASS", pass_count)
                    with col3:
                        fail_count = len(results) - pass_count
                        st.metric("Predicted FAIL", fail_count)
                    with col4:
                        avg_conf = np.mean([r['confidence'] for r in results])
                        st.metric("Avg Confidence", f"{avg_conf:.1f}%")
                    
                    st.divider()
                    
                    st.subheader("📊 Results Table")
                    st.dataframe(results_df, use_container_width=True)
                    
                    st.divider()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("💾 Export CSV"):
                            filepath = DataExporter.export_batch_predictions(results)
                            st.success(f"✅ Exported to {filepath}")
                    
                    with col2:
                        if st.button("📊 Export JSON"):
                            filepath = DataExporter.export_prediction(
                                {"batch_results": results}, 
                                format="json"
                            )
                            st.success(f"✅ Exported to {filepath}")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")


def show_model_performance(trainer, engineer):
    st.header("📈 Model Performance Analysis")
    
    st.subheader("🎯 Model Metrics Comparison")
    
    metrics_df = pd.DataFrame(trainer.metrics).T
    st.dataframe(metrics_df, use_container_width=True)
    
    viz_engine = VisualizationEngine()
    fig = viz_engine.create_metrics_comparison(trainer.metrics)
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    st.subheader("🔍 Feature Importance Analysis")
    
    if hasattr(trainer.best_model, 'feature_importances_'):
        importance = dict(zip(trainer.feature_names, trainer.best_model.feature_importances_))
        fig = viz_engine.create_feature_importance_chart(importance, top_n=15)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("📊 All Features")
        importance_df = pd.DataFrame({
            'Feature': list(importance.keys()),
            'Importance': list(importance.values())
        }).sort_values('Importance', ascending=False)
        st.dataframe(importance_df, use_container_width=True)
    
    st.divider()
    
    st.subheader("📄 Generate Competition Report")
    
    if st.button("📝 Generate Paper-Ready Results"):
        formatter = CompetitionFormatter()
        
        feature_importance = None
        if hasattr(trainer.best_model, 'feature_importances_'):
            feature_importance = dict(zip(trainer.feature_names, 
                                         trainer.best_model.feature_importances_))
        
        report = formatter.generate_paper_results(trainer.metrics, feature_importance)
        
        st.json(report)
        
        st.divider()
        
        st.subheader("📊 LaTeX Table")
        latex = formatter.generate_latex_table(trainer.metrics)
        st.code(latex, language="latex")


def show_feedback_system():
    st.header("🔄 Feedback & Self-Correction System")
    
    memory = FeedbackMemory()
    
    stats = memory.get_statistics()
    
    if stats:
        st.subheader("📊 System Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Corrections", stats.get('total_corrections', 0))
        with col2:
            st.metric("Correct Predictions", stats.get('correct_predictions', 0))
        with col3:
            acc = stats.get('accuracy', 0)
            st.metric("Real-World Accuracy", f"{acc:.2%}" if acc else "N/A")
    
    st.divider()
    
    st.subheader("➕ Add Correction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        student_id = st.text_input("Student ID", "STU_0001")
    with col2:
        actual_result = st.selectbox("Actual Result", ["PASS", "FAIL"])
    
    if st.button("✅ Submit Correction"):
        correction = memory.add_correction(student_id, actual_result)
        if correction:
            st.success("✅ Correction recorded!")
            st.json(correction)
        else:
            st.warning("⚠️ No prediction found for this student ID")
    
    st.divider()
    
    st.subheader("📜 Prediction History")
    
    history = memory.get_prediction_history(limit=20)
    
    if history:
        history_df = pd.DataFrame(history)
        if 'timestamp' in history_df.columns:
            history_df = history_df[['id', 'timestamp', 'prediction', 'actual_result', 'corrected']]
        st.dataframe(history_df, use_container_width=True)
    else:
        st.info("No predictions in history yet")


def show_settings_export(trainer, engineer):
    st.header("⚙️ Settings & Export")
    
    st.subheader("📧 Email Alert Configuration")
    
    email_enabled = st.checkbox("Enable Email Alerts")
    
    if email_enabled:
        sender_email = st.text_input("Sender Email (Gmail)")
        sender_password = st.text_input("App Password", type="password")
        recipient_email = st.text_input("Recipient Email")
        
        if st.button("💾 Save Email Config"):
            st.session_state['email_config'] = {
                'sender': sender_email,
                'password': sender_password,
                'recipient': recipient_email
            }
            st.success("✅ Email configuration saved!")
    
    st.divider()
    
    st.subheader("💾 Model Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Save Current Model"):
            filepath = ModelPersistence.save(trainer, engineer)
            st.success(f"✅ Model saved to {filepath}")
    
    with col2:
        if st.button("📊 Export Model Report"):
            filepath = DataExporter.export_model_report(trainer.metrics)
            st.success(f"✅ Report exported to {filepath}")
    
    st.divider()
    
    st.subheader("📁 Data Management")
    
    if st.button("🗑️ Clear Feedback Memory"):
        if st.button("⚠️ Confirm Delete", type="secondary"):
            memory = FeedbackMemory()
            memory.memory = {"predictions": [], "corrections": [], "statistics": {}}
            memory._save()
            st.success("✅ Feedback memory cleared!")


def create_feature_vector(student_dict):
    features = []
    
    features.extend([
        student_dict["age"],
        student_dict["studytime"],
        student_dict["failures"],
        student_dict["absences"],
        student_dict["G1"],
        student_dict["G2"]
    ])
    
    features.extend([
        1 if student_dict["school"] == "GP" else 0,
        1 if student_dict["sex"] == "F" else 0,
        1 if student_dict["address"] == "U" else 0,
        1 if student_dict["famsize"] == "GT3" else 0,
        1 if student_dict["Pstatus"] == "T" else 0,
        student_dict["Medu"],
        student_dict["Fedu"],
        1 if student_dict["schoolsup"] == "yes" else 0,
        1 if student_dict["famsup"] == "yes" else 0,
        1 if student_dict["paid"] == "yes" else 0,
        1 if student_dict["activities"] == "yes" else 0,
        1 if student_dict["nursery"] == "yes" else 0,
        1 if student_dict["higher"] == "yes" else 0,
        1 if student_dict["internet"] == "yes" else 0,
        1 if student_dict["romantic"] == "yes" else 0,
    ])
    
    parent_edu_avg = (student_dict["Medu"] + student_dict["Fedu"]) / 2
    parent_edu_max = max(student_dict["Medu"], student_dict["Fedu"])
    total_support = (1 if student_dict["schoolsup"] == "yes" else 0) + (1 if student_dict["famsup"] == "yes" else 0)
    study_quality = student_dict["studytime"] * (5 - student_dict["failures"])
    academic_trend = student_dict["G2"] - student_dict["G1"]
    avg_grade = (student_dict["G1"] + student_dict["G2"]) / 2
    grade_volatility = abs(student_dict["G2"] - student_dict["G1"])
    
    family_quality = parent_edu_avg + 3 + total_support
    social_score = 3 + 1 + 1
    health_attendance = 3 - (student_dict["absences"] / 10)
    
    features.extend([
        parent_edu_avg,
        parent_edu_max,
        total_support,
        study_quality,
        academic_trend,
        avg_grade,
        grade_volatility,
        family_quality,
        social_score,
        health_attendance
    ])
    
    return features


if __name__ == "__main__":
    main()