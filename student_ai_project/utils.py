import pandas as pd
import numpy as np
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class DataExporter:
    
    @staticmethod
    def export_prediction(prediction_data, format="json", filepath=None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if filepath is None:
            Path("exports").mkdir(exist_ok=True)
            filepath = f"exports/prediction_{timestamp}.{format}"
        
        if format == "json":
            with open(filepath, 'w') as f:
                json.dump(prediction_data, f, indent=2)
        
        elif format == "csv":
            df = pd.DataFrame([prediction_data])
            df.to_csv(filepath, index=False)
        
        elif format == "txt":
            with open(filepath, 'w') as f:
                f.write("STUDENT PERFORMANCE PREDICTION REPORT\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Prediction: {prediction_data.get('result', 'N/A')}\n")
                f.write(f"Confidence: {prediction_data.get('confidence', 0):.2f}%\n\n")
                
                if 'probabilities' in prediction_data:
                    f.write("Probabilities:\n")
                    for key, val in prediction_data['probabilities'].items():
                        f.write(f"  {key.upper()}: {val:.2%}\n")
                
                f.write("\n" + "=" * 50 + "\n")
        
        return filepath
    
    @staticmethod
    def export_batch_predictions(predictions_list, filepath="exports/batch_predictions.csv"):
        Path(filepath).parent.mkdir(exist_ok=True)
        
        df = pd.DataFrame(predictions_list)
        df.to_csv(filepath, index=False)
        
        return filepath
    
    @staticmethod
    def export_model_report(metrics, filepath="exports/model_report.json"):
        Path(filepath).parent.mkdir(exist_ok=True)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "summary": {
                "best_model": max(metrics.items(), key=lambda x: x[1]["roc_auc"])[0],
                "best_accuracy": max(m["accuracy"] for m in metrics.values()),
                "best_f1": max(m["f1"] for m in metrics.values()),
                "best_roc_auc": max(m["roc_auc"] for m in metrics.values())
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        return filepath


class EmailAlertSystem:
    
    def __init__(self, smtp_server="smtp.gmail.com", smtp_port=587):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = None
        self.sender_password = None
    
    def configure(self, sender_email, sender_password):
        self.sender_email = sender_email
        self.sender_password = sender_password
    
    def send_alert(self, recipient_email, student_data, prediction_result, hybrid_decision):
        if not self.sender_email or not self.sender_password:
            return {"status": "error", "message": "Email not configured"}
        
        subject = f"⚠️ Student Alert: {hybrid_decision['final_decision']}"
        
        body = f"""
        STUDENT PERFORMANCE ALERT
        ========================
        
        Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        
        AI PREDICTION:
        - Result: {prediction_result['result']}
        - Confidence: {prediction_result['confidence']:.1f}%
        - Pass Probability: {prediction_result['probabilities']['pass']:.1%}
        
        DECISION SYSTEM:
        - Final Decision: {hybrid_decision['final_decision']}
        - Risk Score: {hybrid_decision['risk_score']}/100
        - Recommendation: {hybrid_decision['recommendation']}
        
        FLAGS:
        """
        
        for flag in hybrid_decision['flags']:
            body += f"  • [{flag['type']}] {flag['message']}\n"
        
        body += "\n\nThis is an automated alert. Please review the student's case."
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = recipient_email
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            return {"status": "success", "message": "Alert sent successfully"}
        
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def send_batch_summary(self, recipient_email, batch_results):
        if not self.sender_email or not self.sender_password:
            return {"status": "error", "message": "Email not configured"}
        
        high_risk = sum(1 for r in batch_results if r.get('risk_score', 0) >= 50)
        total = len(batch_results)
        
        subject = f"📊 Batch Analysis Summary: {high_risk}/{total} High Risk Students"
        
        body = f"""
        BATCH PREDICTION SUMMARY
        ========================
        
        Total Students Analyzed: {total}
        High Risk Cases: {high_risk}
        Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        
        BREAKDOWN:
        """
        
        for i, result in enumerate(batch_results[:10], 1):
            body += f"\n{i}. Student {result.get('student_id', 'N/A')}\n"
            body += f"   Prediction: {result.get('result', 'N/A')} ({result.get('confidence', 0):.1f}% confidence)\n"
            body += f"   Risk Score: {result.get('risk_score', 0)}/100\n"
        
        if total > 10:
            body += f"\n... and {total - 10} more students\n"
        
        body += "\n\nDetailed report attached separately."
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = recipient_email
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            return {"status": "success", "message": "Summary sent successfully"}
        
        except Exception as e:
            return {"status": "error", "message": str(e)}


class VisualizationEngine:
    
    @staticmethod
    def create_confusion_matrix(y_true, y_pred):
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted FAIL', 'Predicted PASS'],
            y=['Actual FAIL', 'Actual PASS'],
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 20}
        ))
        
        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted',
            yaxis_title='Actual',
            width=500,
            height=500
        )
        
        return fig
    
    @staticmethod
    def create_feature_importance_chart(feature_importance, top_n=15):
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        features, importances = zip(*sorted_features)
        
        fig = go.Figure(go.Bar(
            x=list(importances),
            y=list(features),
            orientation='h',
            marker=dict(color=list(importances), colorscale='Viridis')
        ))
        
        fig.update_layout(
            title=f'Top {top_n} Feature Importance',
            xaxis_title='Importance',
            yaxis_title='Feature',
            height=500,
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def create_probability_distribution(probabilities_list):
        df = pd.DataFrame(probabilities_list)
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df['pass'], name='Pass Probability', opacity=0.7))
        fig.add_trace(go.Histogram(x=df['fail'], name='Fail Probability', opacity=0.7))
        
        fig.update_layout(
            title='Prediction Probability Distribution',
            xaxis_title='Probability',
            yaxis_title='Count',
            barmode='overlay',
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_roc_curve(y_true, y_proba):
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', 
                                 name=f'ROC (AUC = {roc_auc:.3f})',
                                 line=dict(color='blue', width=2)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                 name='Random Classifier',
                                 line=dict(color='red', dash='dash')))
        
        fig.update_layout(
            title='ROC Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=500,
            width=600
        )
        
        return fig
    
    @staticmethod
    def create_what_if_comparison(what_if_results):
        scenarios = what_if_results['scenarios'][:8]
        
        descriptions = [s['description'] for s in scenarios]
        impacts = [s['impact'] * 100 for s in scenarios]
        colors = ['green' if i > 0 else 'red' for i in impacts]
        
        fig = go.Figure(go.Bar(
            x=impacts,
            y=descriptions,
            orientation='h',
            marker=dict(color=colors),
            text=[f"{i:+.1f}%" for i in impacts],
            textposition='outside'
        ))
        
        fig.update_layout(
            title='What-If Scenario Impact',
            xaxis_title='Change in Pass Probability (%)',
            yaxis_title='Scenario',
            height=400,
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def create_metrics_comparison(metrics):
        models = list(metrics.keys())
        accuracy = [metrics[m]['accuracy'] for m in models]
        f1 = [metrics[m]['f1'] for m in models]
        roc_auc = [metrics[m]['roc_auc'] for m in models]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Accuracy', x=models, y=accuracy))
        fig.add_trace(go.Bar(name='F1 Score', x=models, y=f1))
        fig.add_trace(go.Bar(name='ROC AUC', x=models, y=roc_auc))
        
        fig.update_layout(
            title='Model Performance Comparison',
            xaxis_title='Model',
            yaxis_title='Score',
            barmode='group',
            height=500
        )
        
        return fig
    
    @staticmethod
    def create_risk_gauge(risk_score):
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Risk Score"},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 50], 'color': "yellow"},
                    {'range': [50, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        
        fig.update_layout(height=400)
        return fig


class BatchProcessor:
    
    def __init__(self, model_trainer, feature_engineer):
        self.model_trainer = model_trainer
        self.feature_engineer = feature_engineer
    
    def process_csv(self, filepath):
        df = pd.read_csv(filepath, sep=";")
        X, y, df_processed = self.feature_engineer.create_features(df)
        
        results = []
        for idx, row in X.iterrows():
            student_data = row.values.tolist()
            prediction = self.model_trainer.predict_with_confidence(student_data)
            
            result = {
                "student_id": f"STU_{idx:04d}",
                "result": prediction["result"],
                "confidence": prediction["confidence"],
                "pass_probability": prediction["probabilities"]["pass"],
                "fail_probability": prediction["probabilities"]["fail"]
            }
            results.append(result)
        
        return results
    
    def process_students(self, students_list):
        results = []
        
        for student in students_list:
            prediction = self.model_trainer.predict_with_confidence(student["features"])
            
            result = {
                "student_id": student.get("id", "UNKNOWN"),
                "result": prediction["result"],
                "confidence": prediction["confidence"],
                "probabilities": prediction["probabilities"]
            }
            results.append(result)
        
        return results


class CompetitionFormatter:
    
    @staticmethod
    def generate_paper_results(metrics, feature_importance, cv_scores=None):
        report = {
            "title": "Student Performance Prediction using Ensemble Learning",
            "abstract": "Machine learning system for predicting student academic outcomes",
            "methodology": {
                "models": list(metrics.keys()),
                "features": len(feature_importance) if feature_importance else 0,
                "validation": "5-fold cross-validation with stratified sampling"
            },
            "results": {
                "model_performance": metrics,
                "best_model": max(metrics.items(), key=lambda x: x[1]["roc_auc"])[0],
                "cross_validation": cv_scores if cv_scores else {}
            },
            "feature_analysis": {
                "top_features": dict(sorted(feature_importance.items(), 
                                           key=lambda x: x[1], 
                                           reverse=True)[:10]) if feature_importance else {}
            }
        }
        
        return report
    
    @staticmethod
    def export_kaggle_format(predictions, filepath="submission.csv"):
        df = pd.DataFrame(predictions)
        df.to_csv(filepath, index=False)
        return filepath
    
    @staticmethod
    def generate_latex_table(metrics):
        latex = "\\begin{table}[h]\n\\centering\n"
        latex += "\\begin{tabular}{|l|c|c|c|}\n\\hline\n"
        latex += "Model & Accuracy & F1-Score & ROC-AUC \\\\\n\\hline\n"
        
        for model, scores in metrics.items():
            latex += f"{model} & {scores['accuracy']:.4f} & {scores['f1']:.4f} & {scores['roc_auc']:.4f} \\\\\n"
        
        latex += "\\hline\n\\end{tabular}\n"
        latex += "\\caption{Model Performance Comparison}\n"
        latex += "\\label{tab:model_performance}\n\\end{table}"
        
        return latex