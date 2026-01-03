import pandas as pd
from core_model import FeatureEngineer, ModelTrainer, ModelPersistence


def train_model():
    print("=" * 60)
    print("🎓 STUDENT PERFORMANCE PREDICTION - MODEL TRAINING")
    print("=" * 60)
    
    print("\n[1/4] Loading dataset...")
    try:
        df = pd.read_csv("data/student-mat.csv", sep=";")
        print(f"✅ Loaded {len(df)} student records")
        print(f"    Columns: {len(df.columns)}")
    except FileNotFoundError:
        print("❌ ERROR: data/student-mat.csv not found!")
        print("\n📥 Download from:")
        print("   https://archive.ics.uci.edu/ml/datasets/Student+Performance")
        print("\n📁 Place it in: data/student-mat.csv")
        return
    except Exception as e:
        print(f"❌ ERROR loading data: {e}")
        return
    
    print("\n[2/4] Engineering features...")
    engineer = FeatureEngineer()
    X, y, df_processed = engineer.create_features(df)
    print(f"✅ Created {len(engineer.feature_names)} features")
    print(f"    PASS: {sum(y == 1)} students ({sum(y == 1)/len(y)*100:.1f}%)")
    print(f"    FAIL: {sum(y == 0)} students ({sum(y == 0)/len(y)*100:.1f}%)")
    
    print("\n[3/4] Training models (this may take 30-60 seconds)...")
    trainer = ModelTrainer()
    results = trainer.train(X, y, engineer.feature_names)
    
    print(f"✅ Training complete!")
    print(f"    Train size: {results['train_size']}")
    print(f"    Test size: {results['test_size']}")
    print(f"    Best model: {results['best_model']}")
    
    print("\n📊 MODEL PERFORMANCE:")
    print("-" * 60)
    for model_name, metrics in results['metrics'].items():
        print(f"\n{model_name}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    print("\n[4/4] Saving model...")
    filepath = ModelPersistence.save(trainer, engineer)
    print(f"✅ Model saved to: {filepath}")
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE! Model ready to use.")
    print("=" * 60)
    print("\n🚀 Next steps:")
    print("   1. Run: streamlit run app.py")
    print("   2. Start making predictions!")
    print("=" * 60)


if __name__ == "__main__":
    train_model()