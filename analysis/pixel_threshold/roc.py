import numpy as np
import pandas as pd
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns

# Configuration
scene_numbers = [40753679, 47333462, 47333473]
queries = ['ball','door','floor','painting','sofa','table','wall','window']
num_images = [80, 86, 78]  # [80, 86, 78]
image_directory = Path('Validation/')

crop_top = 86
crop_bottom = 86
crop_left = 15
crop_right = 15

# Thresholds to evaluate
thresholds = [0.01, 0.1, 0.5, 1.0, 2.0]
color_threshold = 0.85

# def is_pixel_above_threshold(rgb_pixel):
#     """Check if pixel meets the color threshold criteria"""
#     r, g, b = rgb_pixel

#     if color_threshold == 0.8:
#         is_red_high = r > 120
#         is_green_high = g > 210
#         is_blue_low = b < 85

#     elif color_threshold == 0.7:
#         is_red_high = r > 65
#         is_green_high = g > 190
#         is_blue_low = b < 115

#     elif color_threshold == 0.9:
#         is_red_high = r > 190
#         is_green_high = g > 225
#         is_blue_low = b < 35

#     return is_red_high and is_green_high and is_blue_low

def process_image(image_path, filename):
    """Process a single image and return pixel statistics"""
    img = cv2.imread(str(image_path))
    if img is None:
        return None, None, None
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    cropped = img_rgb[crop_top:h-crop_bottom, crop_left:w-crop_right]
    
    # Save cropped image for verification
    # p = Path("cropped")
    # p.mkdir(exist_ok=True)
    # cropped_path = p / f"{filename}_cropped.png"
    # cv2.imwrite(str(cropped_path), cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))

    crop_h, crop_w = cropped.shape[:2]
    total_pixels = crop_h * crop_w
    
    # Count pixels above threshold
    pixels_above = 0
    # for i in range(crop_h):
    #     for j in range(crop_w):
    #         if is_pixel_above_threshold(cropped[i, j]):
    #             pixels_above += 1
    
    r = cropped[:, :, 0]
    g = cropped[:, :, 1]
    b = cropped[:, :, 2]
    
    if color_threshold == 0.7:
        mask = (r > 65) & (g > 190) & (b < 115)
    if color_threshold == 0.8:
        mask = (r > 120) & (g > 210) & (b < 85)
    elif color_threshold == 0.85:
        mask = (r > 155) & (g > 217) & (b < 60)
    elif color_threshold == 0.9:
        mask = (r > 190) & (g > 225) & (b < 35)
    else:
        # Fallback to the 0.8 rule for unknown thresholds
        mask = (r > 120) & (g > 210) & (b < 85)
    pixels_above = np.count_nonzero(mask)
    
    percentage_above = (pixels_above / total_pixels) * 100
    return total_pixels, pixels_above, percentage_above

def load_cev(csv_path):
    """
    Load ground truth labels from CSV file.
    Expected format: Scene No, Query, Image Index, True Label
    where True Label is 1 (present) or 0 (not present)
    """
    try:
        gt_df = pd.read_csv(csv_path)
        return gt_df
    except FileNotFoundError:
        print(f"Warning: Ground truth file '{csv_path}' not found.")
        print("Creating synthetic ground truth for demonstration...")
        return None

def analyze_images():
    """Process all images and collect statistics"""
    results = []
    
    for scene_idx, scene_no in enumerate(scene_numbers):
        for query in queries:
            for i in range(num_images[scene_idx]):
                image_path = f"Validation/{scene_no}_{query}_{i}_rendered.png"
                filename = f"{scene_no}_{query}_{i}_rendered.png"

                if not Path(image_path).exists():
                    print(f"Warning: {image_path} not found, skipping...")
                    continue

                print(f"Processing: {image_path}")

                total_pixels, pixels_above, percentage_above = process_image(
                    image_path, filename
                )
                
                if total_pixels is None:
                    print(f"Error processing {image_path}")
                    continue
                
                results.append({
                    'Scene No': scene_no,
                    'Query': query,
                    'Image Index': i,
                    'Filename': image_path,
                    'Total Pixels': total_pixels,
                    'Pixels Above Threshold': pixels_above,
                    'Percentage Above': round(percentage_above, 4),
                })
    
    return pd.DataFrame(results)

def generate_roc_curves(df, ground_truth_df, color_threshold):
    """Generate ROC curves for different thresholds"""
    
    # If no ground truth provided, create synthetic data for demonstration
    # if ground_truth_df is None:
    #     print("\n⚠️  No ground truth provided. Creating synthetic labels...")
    #     print("Note: For real analysis, provide a ground_truth.csv file\n")
        
    #     # Synthetic rule: if query matches certain keywords and percentage > 2%, it's a match
    #     relevant_queries = ['door', 'floor', 'sofa', 'table', 'wall']
    #     df['True Label'] = df.apply(
    #         lambda row: 1 if (row['Query'] in relevant_queries and row['Percentage Above'] > 2) 
    #         or (row['Query'] not in relevant_queries and row['Percentage Above'] > 5)
    #         else 0, axis=1
    #     )
    # else:
    #     # Merge with ground truth
    df = df.merge(ground_truth_df, on=['Scene No', 'Query', 'Image Index'], how='left')
    if 'True Label' not in df.columns:
        raise ValueError("Ground truth CSV must have 'True Label' column")
    
    # Calculate metrics for each threshold
    metrics_results = []
    
    for threshold in thresholds:
        df[f'Predicted_{threshold}'] = (df['Percentage Above'] >= threshold).astype(int)
        
        # Calculate confusion matrix
        cm = confusion_matrix(df['True Label'], df[f'Predicted_{threshold}'])
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # Handle edge cases where not all classes are present
            tp = fp = tn = fn = 0
            if cm.shape == (1, 1):
                if df['True Label'].iloc[0] == 1:
                    tp = cm[0, 0]
                else:
                    tn = cm[0, 0]
        
        # Calculate metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics_results.append({
            'Threshold (%)': threshold,
            'TP': tp,
            'FP': fp,
            'TN': tn,
            'FN': fn,
            'Accuracy': round(accuracy, 4),
            'Precision': round(precision, 4),
            'Recall': round(recall, 4),
            'F1-Score': round(f1, 4)
        })
    
    metrics_df = pd.DataFrame(metrics_results)
    
    # Generate ROC curve
    fpr, tpr, roc_thresholds = roc_curve(df['True Label'], df['Percentage Above'])
    roc_auc = auc(fpr, tpr)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # ROC Curve
    axes[0, 0].plot(fpr, tpr, color='darkorange', lw=2, 
                     label=f'ROC curve (AUC = {roc_auc:.4f})')
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('ROC Curve')
    axes[0, 0].legend(loc="lower right")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].scatter(fpr, tpr, s=30, alpha=0.3)

    # Precision-Recall vs Threshold
    axes[0, 1].plot(metrics_df['Threshold (%)'], metrics_df['Precision'], 
                     'o-', label='Precision', linewidth=2)
    axes[0, 1].plot(metrics_df['Threshold (%)'], metrics_df['Recall'], 
                     's-', label='Recall', linewidth=2)
    axes[0, 1].plot(metrics_df['Threshold (%)'], metrics_df['F1-Score'], 
                     '^-', label='F1-Score', linewidth=2)
    axes[0, 1].set_xlabel('Threshold (%)')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_title('Performance Metrics vs Threshold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Confusion Matrix for best F1 threshold
    best_idx = metrics_df['F1-Score'].idxmax()
    best_threshold = metrics_df.loc[best_idx, 'Threshold (%)']
    
    cm = confusion_matrix(df['True Label'], df[f'Predicted_{best_threshold}'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    axes[1, 0].set_title(f'Confusion Matrix (Threshold = {best_threshold}%)')
    axes[1, 0].set_ylabel('True Label')
    axes[1, 0].set_xlabel('Predicted Label')
    
    # Distribution of percentages by true label
    positive_samples = df[df['True Label'] == 1]['Percentage Above']
    negative_samples = df[df['True Label'] == 0]['Percentage Above']
    
    axes[1, 1].hist(negative_samples, bins=30, alpha=0.6, label='Negative', color='red')
    axes[1, 1].hist(positive_samples, bins=30, alpha=0.6, label='Positive', color='green')
    axes[1, 1].set_xlabel('Percentage Above Threshold')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Pixel Percentages')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'roc_analysis_{color_threshold}.png', dpi=300, bbox_inches='tight')
    print(f"✓ ROC analysis plot saved to: roc_analysis.png")
    
    return df, metrics_df, roc_auc

def main():
    """Main execution function"""
    print("="*60)
    print("SEMANTIC SIMILARITY ANALYSIS WITH ROC CURVES")
    print("="*60)
    
    # Process all images
    print("\n1. Processing images...")
    df = analyze_images()
    
    print(f"\n✓ Color threshold set to: {color_threshold}")
    # Save detailed results
    output_file = f'semantic_similarity_analysis_{color_threshold}.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✓ Detailed results saved to: {output_file}")
    
    # Load ground truth (if available)
    ground_truth = load_cev("gt.csv")
    #df = load_cev(output_file)
    
    # Generate ROC curves and metrics
    print("\n2. Generating ROC curves and performance metrics...")
    df_with_labels, metrics_df, roc_auc = generate_roc_curves(df, ground_truth, color_threshold)
    
    # Save metrics
    metrics_file = f'threshold_metrics_{color_threshold}.csv'
    metrics_df.to_csv(metrics_file, index=False)
    print(f"✓ Threshold metrics saved to: {metrics_file}")
    
    # Save final results with predictions
    final_output = f'results_with_predictions_{color_threshold}.csv'
    df_with_labels.to_csv(final_output, index=False)
    print(f"✓ Results with predictions saved to: {final_output}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total images processed: {len(df)}")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print(f"\nBest performing threshold: {metrics_df.loc[metrics_df['F1-Score'].idxmax(), 'Threshold (%)']}%")
    print("\nPerformance Metrics by Threshold:")
    print(metrics_df.to_string(index=False))
    print("="*60)

if __name__ == "__main__":
    main()