import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (confusion_matrix, 
                           roc_auc_score, f1_score, accuracy_score, 
                           precision_score, recall_score, roc_curve)
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import warnings
import time
warnings.filterwarnings('ignore')

class SimpleMLComparison:
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        self.training_times = {}
        self.original_data = None  
        
    def load_and_preprocess_data(self, data_path=None):
        if data_path:
            data = pd.read_csv(data_path)
        else:
            # Generate sample data for demonstration
            print("No data path provided. Generating sample dataset...")
            X, y = make_classification(
                n_samples=1000, n_features=20, n_informative=10,
                n_redundant=10, n_clusters_per_class=1, random_state=42
            )
            
            # Convert to DataFrame
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            data = pd.DataFrame(X, columns=feature_names)
            data['target'] = y
        
        # Store original data for correlation analysis
        self.original_data = data.copy()
        
        # Separate features and target
        if 'target' in data.columns:
            target_col = 'target'
        else:
            # Assume last column is target
            target_col = data.columns[-1]
            
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        # Handle categorical variables in features
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        # Handle categorical target variable
        if y.dtype == 'object':
            print(f"Encoding target variable. Original classes: {y.unique()}")
            le_target = LabelEncoder()
            y = le_target.fit_transform(y)
            print(f"Encoded target classes: {np.unique(y)} -> {le_target.classes_}")
        
        # Fill missing values
        X = X.fillna(X.median())
        
        # Ensure binary classification
        if len(np.unique(y)) > 2:
            print("Converting to binary classification...")
            y = (y > np.median(y)).astype(int)
        
        print(f"Dataset shape: {X.shape}")
        print(f"Class distribution: {pd.Series(y).value_counts().to_dict()}")
        
        return X, y
    
    def prepare_data(self, X, y, test_size=0.2):
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Apply SMOTE to balance classes
        smote = SMOTE(random_state=42)
        self.X_train_balanced, self.y_train_balanced = smote.fit_resample(
            self.X_train, self.y_train
        )
        
        # Scale the features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train_balanced)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training samples: {len(self.X_train_scaled)} (after SMOTE)")
        print(f"Test samples: {len(self.X_test_scaled)}")
    
    def train_models(self):
        """Train Random Forest and XGBoost models with timing"""
        print("\nTraining models...")
        
        # Random Forest configurations
        rf_configs = {
            'RF_Default': RandomForestClassifier(random_state=42),
            'RF_Optimized': RandomForestClassifier(
                n_estimators=100, max_depth=10, min_samples_split=5,
                random_state=42
            )
        }
        
        # XGBoost configurations  
        xgb_configs = {
            'XGB_Default': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'XGB_Optimized': xgb.XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                random_state=42, eval_metric='logloss'
            )
        }
        
        # Train all models with timing
        all_configs = {**rf_configs, **xgb_configs}
        
        for name, model in all_configs.items():
            print(f"Training {name}...")
            start_time = time.time()
            model.fit(self.X_train_scaled, self.y_train_balanced)
            training_time = time.time() - start_time
            
            self.models[name] = model
            self.training_times[name] = training_time
    
    def evaluate_models(self):
        """Evaluate all models"""
        print("\nEvaluating models...")
        
        for name, model in self.models.items():
            # Make predictions
            y_pred = model.predict(self.X_test_scaled)
            y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred),
                'recall': recall_score(self.y_test, y_pred),
                'f1_score': f1_score(self.y_test, y_pred),
                'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
                'confusion_matrix': confusion_matrix(self.y_test, y_pred),
                'y_pred_proba': y_pred_proba,  # Store for ROC curve
                'training_time': self.training_times[name]
            }
            
            # Cross-validation F1 score
            cv_f1 = cross_val_score(model, self.X_train_scaled, self.y_train_balanced, 
                                   cv=5, scoring='f1_weighted')
            metrics['cv_f1_mean'] = cv_f1.mean()
            metrics['cv_f1_std'] = cv_f1.std()
            
            self.results[name] = metrics
            
            # Print results
            print(f"\n{name}:")
            print(f"  Accuracy: {metrics['accuracy']:.3f}")
            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall: {metrics['recall']:.3f}")
            print(f"  F1-Score: {metrics['f1_score']:.3f}")
            print(f"  ROC-AUC: {metrics['roc_auc']:.3f}")
            print(f"  CV F1: {metrics['cv_f1_mean']:.3f} ± {metrics['cv_f1_std']:.3f}")
            print(f"  Training Time: {metrics['training_time']:.2f}s")
    
    def plot_correlation_matrix(self):
        """Plot correlation matrix of features"""
        plt.figure(figsize=(12, 10))
        
        # Prepare data for correlation analysis
        if self.original_data is not None:
            data_for_corr = self.original_data.copy()
            
            # Handle categorical variables for correlation
            for col in data_for_corr.columns:
                if data_for_corr[col].dtype == 'object':
                    le = LabelEncoder()
                    data_for_corr[col] = le.fit_transform(data_for_corr[col].astype(str))
            
            # Calculate correlation matrix
            correlation_matrix = data_for_corr.corr()
            
            # Create the heatmap
            sns.heatmap(correlation_matrix, 
                       annot=True, 
                       cmap='coolwarm', 
                       center=0,
                       square=True,
                       fmt='.2f',
                       cbar_kws={'shrink': 0.8})
            
            plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.show()
        else:
            print("No original data available for correlation analysis")
    
    def plot_roc_curves(self):
        """Plot ROC curves for all models in one plot"""
        plt.figure(figsize=(10, 8))
        
        colors = ['blue', 'red', 'green', 'orange']
        
        for i, (name, results) in enumerate(self.results.items()):
            fpr, tpr, _ = roc_curve(self.y_test, results['y_pred_proba'])
            roc_auc = results['roc_auc']
            
            plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                    label=f'{name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.8, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_results(self):
        """Create visualizations"""
        # Set up the plotting style
        plt.style.use('default')
        
        #  Plot correlation matrix first
        self.plot_correlation_matrix()
        
        # Figure 1: Performance metrics and algorithm comparison
        fig1, axes1 = plt.subplots(1, 2, figsize=(15, 6))
        fig1.suptitle('Model Performance Overview', fontsize=16, fontweight='bold')
        
        # 1. Performance metrics comparison
        models = list(self.results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        rf_models = [m for m in models if m.startswith('RF')]
        xgb_models = [m for m in models if m.startswith('XGB')]
        
        # Bar plot of metrics
        x = np.arange(len(metrics))
        width = 0.15
        
        for i, model in enumerate(models):
            values = [self.results[model][metric] for metric in metrics]
            color = 'skyblue' if model.startswith('RF') else 'lightcoral'
            axes1[0].bar(x + i*width, values, width, label=model, color=color, alpha=0.8)
        
        axes1[0].set_xlabel('Metrics')
        axes1[0].set_ylabel('Score')
        axes1[0].set_title('Performance Metrics Comparison')
        axes1[0].set_xticks(x + width * 1.5)
        axes1[0].set_xticklabels(metrics, rotation=45)
        axes1[0].legend()
        axes1[0].grid(True, alpha=0.3)
        
        # 2. Algorithm comparison (RF vs XGB)
        if rf_models and xgb_models:
            rf_f1 = [self.results[m]['f1_score'] for m in rf_models]
            xgb_f1 = [self.results[m]['f1_score'] for m in xgb_models]
            
            axes1[1].boxplot([rf_f1, xgb_f1], labels=['Random Forest', 'XGBoost'])
            axes1[1].set_ylabel('F1-Score')
            axes1[1].set_title('Algorithm Performance Comparison')
            axes1[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Figure 2: Confusion matrices for all models
        n_models = len(models)
        if n_models > 0:
            # Create grid based on number of models
            if n_models <= 2:
                rows, cols = 1, 2
            elif n_models <= 4:
                rows, cols = 2, 2
            else:
                rows = (n_models + 1) // 2
                cols = 2
            
            fig2, axes2 = plt.subplots(rows, cols, figsize=(12, 6*rows))
            fig2.suptitle('Confusion Matrices for All Models', fontsize=16, fontweight='bold')
            
            # Handle subplot indexing
            if n_models == 1:
                axes2 = [axes2]
            elif rows == 1:
                axes2 = [axes2] if n_models == 1 else axes2
            else:
                axes2 = axes2.flatten()
            
            for i, (name, results) in enumerate(self.results.items()):
                if i < len(axes2):
                    cm = results['confusion_matrix']
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes2[i])
                    axes2[i].set_title(f'{name}\nF1: {results["f1_score"]:.3f}')
                    axes2[i].set_xlabel('Predicted')
                    axes2[i].set_ylabel('Actual')
            
            # Hide any unused subplots
            for i in range(n_models, len(axes2)):
                axes2[i].set_visible(False)
            
            plt.tight_layout()
            plt.show()
        
        # Figure 3: ROC Curves for all models
        self.plot_roc_curves()
        
        # Figure 4: Performance vs Training Time
        self.plot_performance_vs_time()
    
    def plot_performance_vs_time(self):
        """Plot F1-Score vs Training Time to show trade-offs"""
        plt.figure(figsize=(10, 6))
        
        models = list(self.results.keys())
        f1_scores = [self.results[model]['f1_score'] for model in models]
        training_times = [self.results[model]['training_time'] for model in models]
        
        # Color by algorithm type
        colors = ['skyblue' if model.startswith('RF') else 'lightcoral' for model in models]
        
        plt.scatter(training_times, f1_scores, c=colors, s=150, alpha=0.7, edgecolors='black')
        
        # Add model labels
        for i, model in enumerate(models):
            plt.annotate(model, (training_times[i], f1_scores[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        plt.xlabel('Training Time (seconds)')
        plt.ylabel('F1-Score')
        plt.title('Model Performance vs Training Time Trade-off')
        plt.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='skyblue', label='Random Forest'),
                          Patch(facecolor='lightcoral', label='XGBoost')]
        plt.legend(handles=legend_elements)
        
        plt.tight_layout()
        plt.show()
    
    def create_summary_table(self):
        """Create an enhanced summary comparison table"""
        print("\n" + "="*90)
        print("MODEL PERFORMANCE SUMMARY")
        print("="*90)
        
        summary_data = []
        for model_name, metrics in self.results.items():
            summary_data.append({
                'Model': model_name,
                'Accuracy': f"{metrics['accuracy']:.3f}",
                'Precision': f"{metrics['precision']:.3f}",
                'Recall': f"{metrics['recall']:.3f}",
                'F1-Score': f"{metrics['f1_score']:.3f}",
                'ROC-AUC': f"{metrics['roc_auc']:.3f}",
                'CV F1': f"{metrics['cv_f1_mean']:.3f} ± {metrics['cv_f1_std']:.3f}",
                'Train Time (s)': f"{metrics['training_time']:.2f}"
            })
        
        df = pd.DataFrame(summary_data)
        df = df.sort_values('F1-Score', ascending=False)
        print(df.to_string(index=False))
        
        print("="*90)

        # Recommendations based on results
        best_f1_model = max(self.results.items(), key=lambda x: x[1]['f1_score'])
        fastest_model = min(self.results.items(), key=lambda x: x[1]['training_time'])
        most_stable = min(self.results.items(), key=lambda x: x[1]['cv_f1_std'])
        
        print(f"\nRECOMMENDATIONS:")
        print(f" Best Overall Performance: {best_f1_model[0]} (F1: {best_f1_model[1]['f1_score']:.3f})")
        print(f" Fastest Training: {fastest_model[0]} ({fastest_model[1]['training_time']:.2f}s)")
        print(f"  Most Stable : {most_stable[0]} (±{most_stable[1]['cv_f1_std']:.3f})")
        
        return df
    
    def run_analysis(self, data_path=None):
        """Run the complete analysis"""
        print("Starting Enhanced ML Model Comparison")
        print("="*50)
        
        # Step 1: Load and preprocess data
        X, y = self.load_and_preprocess_data(data_path)
        
        # Step 2: Prepare data
        self.prepare_data(X, y)
        
        # Step 3: Train models
        self.train_models()
        
        # Step 4: Evaluate models
        self.evaluate_models()
        
        # Step 5: Create visualizations
        self.plot_results()
        
        # Step 6: Create summary
        summary_df = self.create_summary_table()
        
        print("\n Analysis completed successfully!")
        return summary_df

def main():
    analyzer = SimpleMLComparison()
    results = analyzer.run_analysis(data_path=r"C:\Users\sansk\Downloads\archive (3)\adult.csv")  
    return analyzer, results

# Run the analysis
if __name__ == "__main__":
    analyzer, results = main()