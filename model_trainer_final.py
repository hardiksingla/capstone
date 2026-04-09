import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    auc, make_scorer
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FinalBehaviorClassifier:
    """Final model with Logistic Regression + Linear SVM"""
    
    def __init__(self, data_path='data/features/improved_features.csv'):
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.X_scaled = None
        self.scaler = None
        self.models = {}
        self.feature_names = None
        self.cv_results = {}
        self.best_model_name = None
        self.best_model = None
        
    def load_data(self):
        """Load feature data"""
        print("=" * 80)
        print("📂 LOADING DATA")
        print("=" * 80)
        
        self.df = pd.read_csv(self.data_path)
        
        print(f"   Total sessions: {len(self.df)}")
        print(f"   Normal: {sum(self.df['label'] == 0)}")
        print(f"   Suspicious: {sum(self.df['label'] == 1)}")
        
        return self.df
    
    def select_top_features(self, n_features=5):
        """Select only top N most important features"""
        print(f"\n🎯 Selecting top {n_features} features...")
        
        # Train a simple model to get feature importance
        from sklearn.ensemble import RandomForestClassifier
        temp_model = RandomForestClassifier(n_estimators=50, random_state=42)
        temp_model.fit(self.X_scaled, self.y)
        
        # Get top features
        importances = temp_model.feature_importances_
        top_indices = np.argsort(importances)[-n_features:]
        top_features = [self.feature_names[i] for i in top_indices]
        
        print(f"   Selected features: {', '.join(top_features)}")
        
        # Update data
        self.X = self.X.iloc[:, top_indices]
        self.X_scaled = self.scaler.fit_transform(self.X)
        self.feature_names = top_features
    
        return top_features
    
    def prepare_data(self):
        """Prepare features"""
        print("\n" + "=" * 80)
        print("🔨 PREPARING DATA")
        print("=" * 80)
        
        # Drop non-feature columns
        feature_cols = [col for col in self.df.columns 
                       if col not in ['session_id', 'label', 'suspicion_type', 'duration_seconds']]
        
        self.X = self.df[feature_cols]
        self.y = self.df['label']
        self.feature_names = feature_cols
        
        # Handle NaN/inf
        self.X = self.X.replace([np.inf, -np.inf], np.nan)
        self.X = self.X.fillna(0)
        
        print(f"   Features: {len(feature_cols)}")
        print(f"   Samples: {len(self.X)}")
        
        # Feature scaling
        print("\n   Scaling features...")
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        return self.X_scaled, self.y
    
    def define_models(self):
        """Define models to compare"""
        print("\n" + "=" * 80)
        print("🤖 DEFINING MODELS")
        print("=" * 80)
        
        self.models = {
            'Logistic Regression': LogisticRegression(
                penalty='l2',
                C=0.5,
                solver='liblinear',
                class_weight='balanced',
                random_state=42,
                max_iter=1000
            ),
            'Linear SVM': SVC(
                kernel='linear',
                C=1.0,
                class_weight='balanced',
                probability=True,
                random_state=42
            )
        }
        
        for name in self.models.keys():
            print(f"   ✓ {name}")
        
        return self.models
    
    def cross_validate_models(self, n_folds=5):
        """Perform 5-fold cross-validation"""
        print("\n" + "=" * 80)
        print(f"📊 {n_folds}-FOLD CROSS-VALIDATION")
        print("=" * 80)
        
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1',
            'roc_auc': 'roc_auc'
        }
        
        for model_name, model in self.models.items():
            print(f"\n   🔍 Evaluating {model_name}...")
            
            # Cross-validation scores
            cv_scores = cross_validate(
                model, self.X_scaled, self.y,
                cv=cv,
                scoring=scoring,
                return_train_score=True,
                n_jobs=-1
            )
            
            # Store results
            self.cv_results[model_name] = cv_scores
            
            # Print results
            print(f"\n   Results (mean ± std):")
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
                train_mean = cv_scores[f'train_{metric}'].mean()
                train_std = cv_scores[f'train_{metric}'].std()
                test_mean = cv_scores[f'test_{metric}'].mean()
                test_std = cv_scores[f'test_{metric}'].std()
                
                print(f"     {metric:12s}: Train {train_mean:.3f} ± {train_std:.3f} | "
                      f"Test {test_mean:.3f} ± {test_std:.3f}")
        
        return self.cv_results
    
    def select_best_model(self):
        """Select best model based on F1 score"""
        print("\n" + "=" * 80)
        print("🏆 SELECTING BEST MODEL")
        print("=" * 80)
        
        best_f1 = 0
        best_name = None
        
        for model_name, scores in self.cv_results.items():
            mean_f1 = scores['test_f1'].mean()
            if mean_f1 > best_f1:
                best_f1 = mean_f1
                best_name = model_name
        
        self.best_model_name = best_name
        self.best_model = self.models[best_name]
        
        print(f"\n  Best Model: {best_name}")
        print(f"   Test F1-Score: {best_f1:.3f}")
        
        # Train on full dataset
        print(f"\n   Training {best_name} on full dataset...")
        self.best_model.fit(self.X_scaled, self.y)
        
        return self.best_model
    
    def plot_results(self, save_dir='results'):
        """Generate all visualizations"""
        print("\n" + "=" * 80)
        print("GENERATING VISUALIZATIONS")
        print("=" * 80)
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Cross-Validation Score Comparison
        self._plot_cv_comparison(save_dir)
        
        # 2. ROC Curves (using cross_val_predict)
        self._plot_roc_curves(save_dir)
        
        # 3. Confusion Matrices
        self._plot_confusion_matrices(save_dir)
        
        # 4. Feature Importance
        self._plot_feature_importance(save_dir)
        
        # 5. Performance Metrics Summary
        self._plot_metrics_summary(save_dir)
        
        print(f"\n   All plots saved to '{save_dir}/' directory")
    
    def _plot_cv_comparison(self, save_dir):
        """Plot cross-validation score comparison"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        model_names = list(self.models.keys())
        
        # Test scores
        test_data = []
        for model_name in model_names:
            for metric in metrics:
                scores = self.cv_results[model_name][f'test_{metric}']
                for score in scores:
                    test_data.append({
                        'Model': model_name,
                        'Metric': metric.upper().replace('_', '-'),
                        'Score': score
                    })
        
        df_test = pd.DataFrame(test_data)
        
        # Boxplot
        sns.boxplot(data=df_test, x='Metric', y='Score', hue='Model', ax=axes[0])
        axes[0].set_title('Cross-Validation Scores Distribution', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Score', fontweight='bold')
        axes[0].set_xlabel('Metric', fontweight='bold')
        axes[0].legend(title='Model', loc='lower right')
        axes[0].grid(axis='y', alpha=0.3)
        axes[0].set_ylim([0, 1.05])
        
        # Bar plot with error bars
        mean_scores = []
        std_scores = []
        for model_name in model_names:
            model_means = []
            model_stds = []
            for metric in metrics:
                scores = self.cv_results[model_name][f'test_{metric}']
                model_means.append(scores.mean())
                model_stds.append(scores.std())
            mean_scores.append(model_means)
            std_scores.append(model_stds)
        
        x = np.arange(len(metrics))
        width = 0.35
        
        for i, (model_name, means, stds) in enumerate(zip(model_names, mean_scores, std_scores)):
            axes[1].bar(x + i*width, means, width, yerr=stds, 
                       label=model_name, alpha=0.8, capsize=5)
        
        axes[1].set_title('Mean CV Scores with Standard Deviation', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Score', fontweight='bold')
        axes[1].set_xlabel('Metric', fontweight='bold')
        axes[1].set_xticks(x + width / 2)
        axes[1].set_xticklabels([m.upper().replace('_', '-') for m in metrics])
        axes[1].legend(title='Model')
        axes[1].grid(axis='y', alpha=0.3)
        axes[1].set_ylim([0, 1.05])
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/cv_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved cv_comparison.png")
    
    def _plot_roc_curves(self, save_dir):
        """Plot ROC curves for all models"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for model_name, model in self.models.items():
            # Get probabilities using cross_val_predict
            y_proba = cross_val_predict(
                model, self.X_scaled, self.y,
                cv=cv,
                method='predict_proba',
                n_jobs=-1
            )[:, 1]
            
            # Compute ROC curve
            fpr, tpr, _ = roc_curve(self.y, y_proba)
            roc_auc = auc(fpr, tpr)
            
            # Plot
            ax.plot(fpr, tpr, linewidth=2.5, 
                   label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        # Diagonal line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Random Classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('ROC Curves - Linear Typing Detection', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved roc_curves.png")
    
    def _plot_confusion_matrices(self, save_dir):
        """Plot confusion matrices for all models"""
        fig, axes = plt.subplots(1, len(self.models), figsize=(6*len(self.models), 5))
        
        if len(self.models) == 1:
            axes = [axes]
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for idx, (model_name, model) in enumerate(self.models.items()):
            # Get predictions using cross_val_predict
            y_pred = cross_val_predict(
                model, self.X_scaled, self.y,
                cv=cv,
                n_jobs=-1
            )
            
            # Compute confusion matrix
            cm = confusion_matrix(self.y, y_pred)
            
            # Normalize
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Plot
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Normal', 'Suspicious'],
                       yticklabels=['Normal', 'Suspicious'],
                       ax=axes[idx], cbar_kws={'label': 'Count'})
            
            axes[idx].set_title(f'{model_name}\n(Accuracy: {accuracy_score(self.y, y_pred):.3f})', 
                              fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('True Label', fontweight='bold')
            axes[idx].set_xlabel('Predicted Label', fontweight='bold')
            
            # Add normalized percentages as text
            for i in range(2):
                for j in range(2):
                    axes[idx].text(j+0.5, i+0.7, f'({cm_normalized[i,j]:.1%})', 
                                 ha='center', va='center', fontsize=9, color='gray')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved confusion_matrices.png")
    
    def _plot_feature_importance(self, save_dir):
        """Plot feature importance for both models"""
        fig, axes = plt.subplots(1, len(self.models), figsize=(10*len(self.models), 8))
        
        if len(self.models) == 1:
            axes = [axes]
        
        for idx, (model_name, model) in enumerate(self.models.items()):
            # Train on full data
            model.fit(self.X_scaled, self.y)
            
            # Get coefficients
            if hasattr(model, 'coef_'):
                importances = np.abs(model.coef_[0])
            else:
                importances = np.ones(len(self.feature_names))
            
            # Create dataframe
            feature_imp_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False).head(15)
            
            # Plot
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_imp_df)))
            axes[idx].barh(range(len(feature_imp_df)), 
                          feature_imp_df['importance'],
                          color=colors)
            axes[idx].set_yticks(range(len(feature_imp_df)))
            axes[idx].set_yticklabels(feature_imp_df['feature'])
            axes[idx].set_xlabel('Absolute Coefficient Value', fontweight='bold')
            axes[idx].set_title(f'Top 15 Features - {model_name}', 
                              fontsize=12, fontweight='bold')
            axes[idx].invert_yaxis()
            axes[idx].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved feature_importance.png")
    
    def _plot_metrics_summary(self, save_dir):
        """Create comprehensive metrics summary table"""
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare data
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        table_data = []
        
        for model_name in self.models.keys():
            row = [model_name]
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
                mean = self.cv_results[model_name][f'test_{metric}'].mean()
                std = self.cv_results[model_name][f'test_{metric}'].std()
                row.append(f'{mean:.3f} ± {std:.3f}')
            table_data.append(row)
        
        # Create table
        table = ax.table(cellText=table_data,
                        colLabels=['Model'] + metrics,
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.2] + [0.16]*5)
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header
        for i in range(len(metrics) + 1):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(table_data) + 1):
            for j in range(len(metrics) + 1):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#E7E6E6')
        
        plt.title('Cross-Validation Performance Summary (5-Fold)', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.savefig(f'{save_dir}/metrics_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved metrics_summary.png")
    
    def save_model(self, model_dir='models'):
        """Save best model"""
        print("\n" + "=" * 80)
        print("💾 SAVING MODEL")
        print("=" * 80)
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = f'{model_dir}/best_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(self.best_model, f)
        print(f"   ✓ Model saved to {model_path}")
        
        # Save scaler
        scaler_path = f'{model_dir}/feature_scaler.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"   ✓ Scaler saved to {scaler_path}")
        
        # Save feature names
        feature_names_path = f'{model_dir}/feature_names.json'
        with open(feature_names_path, 'w') as f:
            json.dump(self.feature_names, f, indent=2)
        print(f"   ✓ Feature names saved to {feature_names_path}")
        
        # Save metadata
        cv_summary = {}
        for model_name, scores in self.cv_results.items():
            cv_summary[model_name] = {
                metric: {
                    'mean': float(scores[f'test_{metric}'].mean()),
                    'std': float(scores[f'test_{metric}'].std())
                }
                for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
            }
        
        metadata = {
            'model_type': self.best_model_name,
            'training_date': datetime.now().isoformat(),
            'n_features': len(self.feature_names),
            'n_samples': len(self.X),
            'cv_folds': 5,
            'cv_results': cv_summary,
            'best_model_metrics': cv_summary[self.best_model_name]
        }
        
        metadata_path = f'{model_dir}/model_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"   ✓ Metadata saved to {metadata_path}")
        
        return model_path
    
    def generate_report(self, save_dir='results'):
        """Generate text report"""
        print("\n" + "=" * 80)
        print("📄 GENERATING REPORT")
        print("=" * 80)
        
        report_path = f'{save_dir}/model_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("LINEAR TYPING DETECTION - MODEL EVALUATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: {self.data_path}\n")
            f.write(f"Total Samples: {len(self.X)}\n")
            f.write(f"Normal: {sum(self.y == 0)}, Suspicious: {sum(self.y == 1)}\n")
            f.write(f"Features: {len(self.feature_names)}\n")
            f.write(f"Cross-Validation: 5-Fold Stratified\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("MODEL COMPARISON\n")
            f.write("=" * 80 + "\n\n")
            
            for model_name in self.models.keys():
                f.write(f"{model_name}:\n")
                f.write("-" * 40 + "\n")
                for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
                    mean = self.cv_results[model_name][f'test_{metric}'].mean()
                    std = self.cv_results[model_name][f'test_{metric}'].std()
                    f.write(f"  {metric.capitalize():12s}: {mean:.3f} ± {std:.3f}\n")
                f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write(f"BEST MODEL: {self.best_model_name}\n")
            f.write("=" * 80 + "\n\n")
            
            best_metrics = self.cv_results[self.best_model_name]
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
                mean = best_metrics[f'test_{metric}'].mean()
                std = best_metrics[f'test_{metric}'].std()
                f.write(f"{metric.capitalize():12s}: {mean:.3f} ± {std:.3f}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("TOP 15 IMPORTANT FEATURES\n")
            f.write("=" * 80 + "\n\n")
            
            # Get feature importance from best model
            self.best_model.fit(self.X_scaled, self.y)
            if hasattr(self.best_model, 'coef_'):
                importances = np.abs(self.best_model.coef_[0])
                feature_imp = sorted(zip(self.feature_names, importances), 
                                   key=lambda x: x[1], reverse=True)[:15]
                for i, (feat, imp) in enumerate(feature_imp, 1):
                    f.write(f"{i:2d}. {feat:40s}: {imp:.4f}\n")
        
        print(f"   ✓ Report saved to {report_path}")
        return report_path

def main():
    """Main training pipeline"""
    print("\n")
    print("=" * 80)
    print("🎯 LINEAR TYPING DETECTION - FINAL MODEL TRAINING")
    print("=" * 80)
    print("\n")
    
    # Initialize
    classifier = FinalBehaviorClassifier()
    
    # Load data
    classifier.load_data()
    
    # Prepare data
    classifier.prepare_data()

    # classifier.select_top_features(n_features=5)
    
    # Define models
    classifier.define_models()
    
    # Cross-validate
    classifier.cross_validate_models(n_folds=5)
    
    # Select best model
    classifier.select_best_model()
    
    # Generate visualizations
    classifier.plot_results(save_dir='results')
    
    # Save model
    classifier.save_model(model_dir='models')
    
    # Generate report
    classifier.generate_report(save_dir='results')
    
    print("\n" + "=" * 80)
    print("✅ TRAINING PIPELINE COMPLETE!")
    print("=" * 80)
    print("\n📊 Results:")
    print(f"   • Best Model: {classifier.best_model_name}")
    print(f"   • Test F1: {classifier.cv_results[classifier.best_model_name]['test_f1'].mean():.3f}")
    print(f"   • Visualizations: results/")
    print(f"   • Model: models/best_model.pkl")
    print(f"   • Report: results/model_report.txt")
    print("=" * 80 + "\n")

if __name__ == '__main__':
    main()