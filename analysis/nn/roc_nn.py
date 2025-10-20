import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns

# Configuration
confidence_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
input_csv = 'nnew_semantic_similarity_analysis.csv'
ground_truth_csv = 'gt.csv'

def load_data(predictions_path, gt_path):
    """Load predictions and ground truth data"""
    try:
        pred_df = pd.read_csv(predictions_path)
        gt_df = pd.read_csv(gt_path)
        
        # Merge predictions with ground truth
        df = pred_df.merge(gt_df, on=['Scene No', 'Query', 'Image Index'], how='left')
        
        if 'True Label' not in df.columns:
            raise ValueError("Ground truth CSV must have 'True Label' column")
        
        # Convert "Object Present" to binary (yes→1, no→0)
        df['Binary Prediction'] = df['Object Present'].map({'yes': 1, 'no': 0})
        
        # Calculate confidence that object IS present (Option A)
        df['Confidence Present'] = df.apply(
            lambda row: row['Confidence'] if row['Object Present'] == 'yes' 
            else 1 - row['Confidence'], 
            axis=1
        )
        
        return df
    
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def evaluate_binary_predictions(df):
    """Part 1: Evaluate the model's binary predictions"""
    print("\n" + "="*60)
    print("PART 1: BINARY PREDICTION EVALUATION")
    print("="*60)
    
    cm = confusion_matrix(df['True Label'], df['Binary Prediction'])
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tp = fp = tn = fn = 0
        if cm.shape == (1, 1):
            if df['True Label'].iloc[0] == 1:
                tp = cm[0, 0]
            else:
                tn = cm[0, 0]
    
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    results = {
        'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
        'Accuracy': round(accuracy, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1-Score': round(f1, 4)
    }
    
    print(f"\nModel's Binary Predictions Performance:")
    print(f"  Accuracy:  {results['Accuracy']}")
    print(f"  Precision: {results['Precision']}")
    print(f"  Recall:    {results['Recall']}")
    print(f"  F1-Score:  {results['F1-Score']}")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {tp}  FP: {fp}")
    print(f"  FN: {fn}  TN: {tn}")
    
    return results, cm

def evaluate_confidence_thresholds(df):
    """Part 2: Evaluate confidence scores across multiple thresholds"""
    print("\n" + "="*60)
    print("PART 2: CONFIDENCE-BASED THRESHOLD EVALUATION")
    print("="*60)
    
    metrics_results = []
    
    for threshold in confidence_thresholds:
        # Predict based on confidence threshold
        df[f'Pred_conf_{threshold}'] = (df['Confidence Present'] >= threshold).astype(int)
        
        cm = confusion_matrix(df['True Label'], df[f'Pred_conf_{threshold}'])
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tp = fp = tn = fn = 0
            if cm.shape == (1, 1):
                if df['True Label'].iloc[0] == 1:
                    tp = cm[0, 0]
                else:
                    tn = cm[0, 0]
        
        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics_results.append({
            'Confidence Threshold': threshold,
            'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
            'Accuracy': round(accuracy, 4),
            'Precision': round(precision, 4),
            'Recall': round(recall, 4),
            'F1-Score': round(f1, 4)
        })
    
    metrics_df = pd.DataFrame(metrics_results)
    
    # Generate ROC curve
    fpr, tpr, roc_thresholds = roc_curve(df['True Label'], df['Confidence Present'])
    roc_auc = auc(fpr, tpr)
    
    # Find optimal threshold (best F1)
    best_idx = metrics_df['F1-Score'].idxmax()
    best_threshold = metrics_df.loc[best_idx, 'Confidence Threshold']
    best_metrics = metrics_df.loc[best_idx]
    
    print(f"\nROC AUC Score: {roc_auc:.4f}")
    print(f"\nOptimal Confidence Threshold: {best_threshold}")
    print(f"  Accuracy:  {best_metrics['Accuracy']}")
    print(f"  Precision: {best_metrics['Precision']}")
    print(f"  Recall:    {best_metrics['Recall']}")
    print(f"  F1-Score:  {best_metrics['F1-Score']}")
    
    return metrics_df, roc_auc, fpr, tpr, best_threshold, df

def visualize_results(df, binary_metrics, binary_cm, conf_metrics_df, 
                     roc_auc, fpr, tpr, best_threshold):
    """Part 3: Create comprehensive visualization"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. ROC Curve
    axes[0, 0].plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.4f})')
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('ROC Curve (Confidence Scores)')
    axes[0, 0].legend(loc="lower right")
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Precision/Recall/F1 vs Threshold
    axes[0, 1].plot(conf_metrics_df['Confidence Threshold'], 
                    conf_metrics_df['Precision'], 'o-', label='Precision', linewidth=2)
    axes[0, 1].plot(conf_metrics_df['Confidence Threshold'], 
                    conf_metrics_df['Recall'], 's-', label='Recall', linewidth=2)
    axes[0, 1].plot(conf_metrics_df['Confidence Threshold'], 
                    conf_metrics_df['F1-Score'], '^-', label='F1-Score', linewidth=2)
    axes[0, 1].axvline(x=best_threshold, color='red', linestyle='--', 
                       label=f'Optimal ({best_threshold})')
    axes[0, 1].set_xlabel('Confidence Threshold')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_title('Performance Metrics vs Confidence Threshold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Confusion Matrix - Binary Predictions
    sns.heatmap(binary_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 2],
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    axes[0, 2].set_title(f"Binary Predictions\n(F1={binary_metrics['F1-Score']:.4f})")
    axes[0, 2].set_ylabel('True Label')
    axes[0, 2].set_xlabel('Predicted Label')
    
    # 4. Confusion Matrix - Best Confidence Threshold
    best_cm = confusion_matrix(df['True Label'], df[f'Pred_conf_{best_threshold}'])
    sns.heatmap(best_cm, annot=True, fmt='d', cmap='Greens', ax=axes[1, 0],
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    best_f1 = conf_metrics_df.loc[conf_metrics_df['Confidence Threshold']==best_threshold, 'F1-Score'].values[0]
    axes[1, 0].set_title(f'Optimal Confidence Threshold ({best_threshold})\n(F1={best_f1:.4f})')
    axes[1, 0].set_ylabel('True Label')
    axes[1, 0].set_xlabel('Predicted Label')
    
    # 5. Distribution of Confidence Present by True Label
    positive_samples = df[df['True Label'] == 1]['Confidence Present']
    negative_samples = df[df['True Label'] == 0]['Confidence Present']
    
    # Transform negative samples to represent probability object is present (1 - confidence)
    neg_present = 1 - negative_samples
    axes[1, 1].hist(neg_present, bins=30, alpha=0.6, label='True Negative', color='red')
    axes[1, 1].hist(positive_samples, bins=30, alpha=0.6, label='True Positive', color='green')
    axes[1, 1].axvline(x=best_threshold, color='black', linestyle='--', 
                       label=f'Optimal Threshold ({best_threshold})')
    axes[1, 1].set_xlabel('Confidence (Object Present)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Confidence Scores')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Comparison Table
    axes[1, 2].axis('off')
    comparison_data = [
        ['Metric', 'Binary Model', 'Optimal Conf.', 'Δ'],
        ['Accuracy', f"{binary_metrics['Accuracy']:.4f}", 
         f"{conf_metrics_df.loc[conf_metrics_df['Confidence Threshold']==best_threshold, 'Accuracy'].values[0]:.4f}",
         f"{conf_metrics_df.loc[conf_metrics_df['Confidence Threshold']==best_threshold, 'Accuracy'].values[0] - binary_metrics['Accuracy']:+.4f}"],
        ['Precision', f"{binary_metrics['Precision']:.4f}", 
         f"{conf_metrics_df.loc[conf_metrics_df['Confidence Threshold']==best_threshold, 'Precision'].values[0]:.4f}",
         f"{conf_metrics_df.loc[conf_metrics_df['Confidence Threshold']==best_threshold, 'Precision'].values[0] - binary_metrics['Precision']:+.4f}"],
        ['Recall', f"{binary_metrics['Recall']:.4f}", 
         f"{conf_metrics_df.loc[conf_metrics_df['Confidence Threshold']==best_threshold, 'Recall'].values[0]:.4f}",
         f"{conf_metrics_df.loc[conf_metrics_df['Confidence Threshold']==best_threshold, 'Recall'].values[0] - binary_metrics['Recall']:+.4f}"],
        ['F1-Score', f"{binary_metrics['F1-Score']:.4f}", 
         f"{best_f1:.4f}",
         f"{best_f1 - binary_metrics['F1-Score']:+.4f}"],
    ]
    
    table = axes[1, 2].table(cellText=comparison_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    axes[1, 2].set_title('Performance Comparison', fontsize=12, weight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('confidence_evaluation_results.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: confidence_evaluation_results.png")

def main():
    """Main execution function"""
    print("="*60)
    print("MODEL CONFIDENCE EVALUATION (OPTION 3)")
    print("="*60)
    
    # Load data
    print(f"\nLoading data from:")
    print(f"  Predictions: {input_csv}")
    print(f"  Ground Truth: {ground_truth_csv}")
    
    df = load_data(input_csv, ground_truth_csv)
    if df is None:
        return
    
    print(f"\n✓ Loaded {len(df)} samples")
    print(f"  Positive samples: {df['True Label'].sum()}")
    print(f"  Negative samples: {len(df) - df['True Label'].sum()}")
    
    # Part 1: Evaluate binary predictions
    binary_metrics, binary_cm = evaluate_binary_predictions(df)
    
    # Part 2: Evaluate confidence thresholds
    conf_metrics_df, roc_auc, fpr, tpr, best_threshold, df = evaluate_confidence_thresholds(df)
    
    # Save metrics
    conf_metrics_df.to_csv('confidence_threshold_metrics.csv', index=False)
    print(f"\n✓ Confidence threshold metrics saved to: confidence_threshold_metrics.csv")
    
    # Save predictions with all confidence thresholds
    df.to_csv('predictions_with_confidence_analysis.csv', index=False)
    print(f"✓ Detailed predictions saved to: predictions_with_confidence_analysis.csv")
    
    # Part 3: Visualize and compare
    print("\n" + "="*60)
    print("PART 3: COMPARISON & VISUALIZATION")
    print("="*60)
    
    visualize_results(df, binary_metrics, binary_cm, conf_metrics_df, 
                     roc_auc, fpr, tpr, best_threshold)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nBinary Model Performance:")
    print(f"  F1-Score: {binary_metrics['F1-Score']:.4f}")
    print(f"\nOptimal Confidence Threshold: {best_threshold}")
    best_f1 = conf_metrics_df.loc[conf_metrics_df['Confidence Threshold']==best_threshold, 'F1-Score'].values[0]
    print(f"  F1-Score: {best_f1:.4f}")
    print(f"\nImprovement: {best_f1 - binary_metrics['F1-Score']:+.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()