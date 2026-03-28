"""
PriDe Evaluation with Comprehensive Visualizations
Generates publication-ready plots comparing baseline vs debiased results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Tuple
import warnings
from pathlib import Path
import argparse
from scipy.stats import chisquare, chi2_contingency

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class PriDeDebiasing:
    """
    PriDe: Debiasing with Prior estimation for multiple choice questions.
    (Class implementation identical to original - keeping it the same)
    """

    def __init__(self, calibration_ratio: float = 0.05,
                 alpha: float = 1.0,
                 random_seed: int = 42):
        self.calibration_ratio = calibration_ratio
        self.alpha = alpha
        self.random_seed = random_seed
        self.global_prior = None
        self.calibrated = False

    def gather_probs(self, observed: np.ndarray, permuted_indices: List) -> List[List[float]]:
        n_options = observed.shape[1]
        gathered_probs = [[] for _ in range(n_options)]
        for pdx, indices in enumerate(permuted_indices):
            for idx, index in enumerate(indices):
                gathered_probs[index].append(observed[pdx, idx])
        return gathered_probs

    def estimate_prior_for_question(self, probs_matrix: np.ndarray,
                                    permuted_indices: List[Tuple]) -> Tuple[np.ndarray, np.ndarray]:
        observed = probs_matrix / (probs_matrix.sum(axis=1, keepdims=True) + 1e-10)
        gathered_probs = self.gather_probs(observed, permuted_indices)
        debiased = np.array([np.mean(probs) for probs in gathered_probs])
        prior = self.softmax(np.log(observed + 1e-10).mean(axis=0))
        return debiased, prior

    def softmax(self, x: np.ndarray) -> np.ndarray:
        x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return x / (np.sum(x, axis=-1, keepdims=True) + 1e-10)

    def split_calibration_test(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        unique_questions = df['question_id'].unique()
        rng = np.random.RandomState(self.random_seed)
        rng.shuffle(unique_questions)
        n_calibration = max(1, int(len(unique_questions) * self.calibration_ratio))
        calibration_questions = unique_questions[:n_calibration]
        test_questions = unique_questions[n_calibration:]
        calibration_df = df[df['question_id'].isin(calibration_questions)].copy()
        test_df = df[df['question_id'].isin(test_questions)].copy()
        return calibration_df, test_df

    def estimate_prior_from_calibration(self, calibration_df: pd.DataFrame) -> np.ndarray:
        question_groups = calibration_df.groupby('question_id')
        all_priors = []
        for qid, group in question_groups:
            group = group.sort_values('permutation_idx')
            probs_matrix = group[['prob_A', 'prob_B', 'prob_C', 'prob_D']].values
            n_options = 4
            permuted_indices = [
                tuple((i + shift) % n_options for i in range(n_options))
                for shift in range(probs_matrix.shape[0])
            ]
            _, prior = self.estimate_prior_for_question(probs_matrix, permuted_indices)
            all_priors.append(prior)
        self.global_prior = np.mean(all_priors, axis=0)
        self.calibrated = True
        return self.global_prior

    def debias_with_prior(self, observed_probs: np.ndarray, prior: np.ndarray) -> np.ndarray:
        debiased = (np.log(observed_probs + 1e-10) -
                   self.alpha * np.log(prior + 1e-10))
        return debiased

    def debias_test_set(self, test_df: pd.DataFrame) -> pd.DataFrame:
        if not self.calibrated:
            raise ValueError("Must estimate prior from calibration set first")
        df_debiased = test_df.copy()
        positions = ['A', 'B', 'C', 'D']
        all_debiased_answers = []
        all_debiased_correct = []
        for idx, row in df_debiased.iterrows():
            observed_probs = np.array([row[f'prob_{pos}'] for pos in positions])
            debiased_logits = self.debias_with_prior(observed_probs, self.global_prior)
            predicted_position = np.argmax(debiased_logits)
            predicted_answer = positions[predicted_position]
            all_debiased_answers.append(predicted_answer)
            is_correct = (predicted_answer == row['correct_position'])
            all_debiased_correct.append(int(is_correct))
        df_debiased['debiased_predicted_answer'] = all_debiased_answers
        df_debiased['debiased_is_correct'] = all_debiased_correct
        return df_debiased

    def fit_and_predict(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        calibration_df, test_df = self.split_calibration_test(df)
        prior = self.estimate_prior_from_calibration(calibration_df)
        test_df_debiased = self.debias_test_set(test_df)
        calibration_info = {
            'n_total_questions': df['question_id'].nunique(),
            'n_calibration_questions': calibration_df['question_id'].nunique(),
            'n_test_questions': test_df['question_id'].nunique(),
            'calibration_samples': len(calibration_df),
            'test_samples': len(test_df),
            'estimated_prior': prior.tolist(),
        }
        return test_df_debiased, calibration_info


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """Load and prepare the CSV data for PriDe analysis."""
    df = pd.read_csv(csv_path)
    for pos in ['A', 'B', 'C', 'D']:
        df[f'prob_{pos}'] = pd.to_numeric(df[f'prob_{pos}'], errors='coerce')
    df['is_correct_fixed'] = (df['predicted_answer'] == df['correct_position']).astype(int)
    return df


def compute_bias_metrics(df: pd.DataFrame, prediction_col: str = 'predicted_answer') -> Dict:
    """Compute all bias metrics for a dataset, including Recall Std from PriDe paper."""
    df_local = df.copy()
    df_local[prediction_col] = df_local[prediction_col].astype(str).str.upper().str.strip()
    df_local['correct_position'] = df_local['correct_position'].astype(str).str.upper().str.strip()
    
    valid_responses = df_local[df_local[prediction_col].isin(["A", "B", "C", "D"])].copy()
    valid_responses["is_correct_eval"] = (
        valid_responses[prediction_col] == valid_responses['correct_position']
    ).astype(int)
    
    choice_counts = valid_responses[prediction_col].value_counts().reindex(["A", "B", "C", "D"], fill_value=0)
    total_valid = len(valid_responses)
    
    # Chi-square test
    expected_per_choice = total_valid / 4
    expected = [expected_per_choice] * 4
    chi2_stat, p_value = chisquare(choice_counts.values, f_exp=expected)
    
    # Position bias score
    choice_percentages = choice_counts.values / total_valid * 100
    position_bias_score = np.std(choice_percentages)
    
    # Recall Standard Deviation (RStd) - from PriDe paper Section 2.2
    # This measures imbalance in recalls across positions A/B/C/D
    positions = ['A', 'B', 'C', 'D']
    recalls = []
    for pos in positions:
        pos_mask = valid_responses['correct_position'] == pos
        if pos_mask.sum() > 0:
            recall = (valid_responses[pos_mask][prediction_col] == pos).mean()
            recalls.append(recall)
        else:
            recalls.append(0.0)
    recall_std = np.std(recalls) * 100  # Report as percentage
    
    # Accuracy by position
    accuracy_by_position = valid_responses.groupby('correct_position')['is_correct_eval'].mean()
    overall_accuracy = valid_responses['is_correct_eval'].mean()
    
    # Accuracy vs position chi-square
    try:
        contingency_table = pd.crosstab(valid_responses['correct_position'], valid_responses['is_correct_eval'])
        chi2_acc, p_acc, _, _ = chi2_contingency(contingency_table)
    except:
        chi2_acc, p_acc = 0, 1
    
    return {
        'choice_counts': choice_counts.to_dict(),
        'choice_percentages': (choice_counts / total_valid * 100).to_dict(),
        'chi2_stat': chi2_stat,
        'chi2_pvalue': p_value,
        'position_bias_score': position_bias_score,
        'recall_std': recall_std,  # NEW: PriDe paper's main bias metric
        'recalls': {pos: recalls[i] for i, pos in enumerate(positions)},  # Individual recalls
        'accuracy_by_position': accuracy_by_position.to_dict(),
        'overall_accuracy': overall_accuracy,
        'chi2_acc_stat': chi2_acc,
        'chi2_acc_pvalue': p_acc,
        'n_samples': total_valid
    }


def create_visualization_folder(dataset_name: str, model_name: str) -> Path:
    """Create organized folder structure for visualizations."""
    base_dir = Path("pride/results/visualizations") / f"{dataset_name}-{model_name}"
    subdirs = {
        'comparison': base_dir / "01_before_after_comparison",
        'distributions': base_dir / "02_choice_distributions",
        'accuracy': base_dir / "03_accuracy_analysis",
        'alpha': base_dir / "04_alpha_selection",
        'detailed': base_dir / "05_detailed_metrics"
    }
    for subdir in subdirs.values():
        subdir.mkdir(parents=True, exist_ok=True)
    return base_dir, subdirs


def plot_choice_distribution_comparison(baseline_metrics: Dict, debiased_metrics: Dict, 
                                        output_path: Path, title_suffix: str = ""):
    """Create side-by-side bar chart comparing choice distributions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    positions = ['A', 'B', 'C', 'D']
    baseline_pcts = [baseline_metrics['choice_percentages'][p] for p in positions]
    debiased_pcts = [debiased_metrics['choice_percentages'][p] for p in positions]
    
    x = np.arange(len(positions))
    width = 0.6
    
    # Baseline
    bars1 = ax1.bar(x, baseline_pcts, width, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
    ax1.axhline(y=25, color='red', linestyle='--', linewidth=2, label='Uniform (25%)')
    ax1.set_xlabel('Position', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Baseline{title_suffix}\nBias Score: {baseline_metrics["position_bias_score"]:.2f}', 
                  fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(positions)
    ax1.legend()
    ax1.set_ylim(0, max(max(baseline_pcts), max(debiased_pcts)) * 1.2)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Debiased
    bars2 = ax2.bar(x, debiased_pcts, width, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
    ax2.axhline(y=25, color='red', linestyle='--', linewidth=2, label='Uniform (25%)')
    ax2.set_xlabel('Position', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax2.set_title(f'After PriDe{title_suffix}\nBias Score: {debiased_metrics["position_bias_score"]:.2f}', 
                  fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(positions)
    ax2.legend()
    ax2.set_ylim(0, max(max(baseline_pcts), max(debiased_pcts)) * 1.2)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_accuracy_by_position_comparison(baseline_metrics: Dict, debiased_metrics: Dict,
                                         output_path: Path, title_suffix: str = ""):
    """Create accuracy by position comparison plot."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    positions = ['A', 'B', 'C', 'D']
    baseline_acc = [baseline_metrics['accuracy_by_position'].get(p, 0) for p in positions]
    debiased_acc = [debiased_metrics['accuracy_by_position'].get(p, 0) for p in positions]
    overall_baseline = baseline_metrics['overall_accuracy']
    overall_debiased = debiased_metrics['overall_accuracy']
    
    x = np.arange(len(positions))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, baseline_acc, width, label='Baseline', color='#FF6B6B', alpha=0.8)
    bars2 = ax.bar(x + width/2, debiased_acc, width, label='After PriDe', color='#4ECDC4', alpha=0.8)
    
    ax.axhline(y=overall_baseline, color='#FF6B6B', linestyle='--', linewidth=2, 
               label=f'Baseline Overall: {overall_baseline:.3f}')
    ax.axhline(y=overall_debiased, color='#4ECDC4', linestyle='--', linewidth=2,
               label=f'PriDe Overall: {overall_debiased:.3f}')
    
    ax.set_xlabel('Correct Answer Position', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(f'Accuracy by Correct Answer Position{title_suffix}\nImprovement: {(overall_debiased - overall_baseline):+.3f}',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(positions)
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(max(baseline_acc), max(debiased_acc)) * 1.2)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels and change indicators
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        change = height2 - height1
        
        ax.text(bar1.get_x() + bar1.get_width()/2., height1,
                f'{height1:.3f}', ha='center', va='bottom', fontsize=9)
        ax.text(bar2.get_x() + bar2.get_width()/2., height2,
                f'{height2:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Add change arrow
        if abs(change) > 0.01:
            color = 'green' if change > 0 else 'red'
            arrow = '↑' if change > 0 else '↓'
            ax.text(x[i], max(height1, height2) * 1.15, f'{arrow}{abs(change):.3f}',
                   ha='center', va='bottom', fontsize=10, color=color, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_metrics_comparison_radar(baseline_metrics: Dict, debiased_metrics: Dict,
                                  output_path: Path, title_suffix: str = ""):
    """Create radar chart comparing multiple metrics."""
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Normalize metrics to 0-1 scale for visualization
    metrics = {
        'Accuracy': (baseline_metrics['overall_accuracy'], debiased_metrics['overall_accuracy']),
        'Uniformity\n(lower bias better)': (
            1 - baseline_metrics['position_bias_score'] / 25,  # Normalize
            1 - debiased_metrics['position_bias_score'] / 25
        ),
        'Chi-square\nUniformity\n(p-value)': (
            baseline_metrics['chi2_pvalue'],
            debiased_metrics['chi2_pvalue']
        ),
        'Position-Accuracy\nIndependence\n(p-value)': (
            baseline_metrics['chi2_acc_pvalue'],
            debiased_metrics['chi2_acc_pvalue']
        ),
    }
    
    categories = list(metrics.keys())
    N = len(categories)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    baseline_values = [metrics[cat][0] for cat in categories]
    debiased_values = [metrics[cat][1] for cat in categories]
    baseline_values += baseline_values[:1]
    debiased_values += debiased_values[:1]
    
    ax.plot(angles, baseline_values, 'o-', linewidth=2, label='Baseline', color='#FF6B6B')
    ax.fill(angles, baseline_values, alpha=0.25, color='#FF6B6B')
    ax.plot(angles, debiased_values, 'o-', linewidth=2, label='After PriDe', color='#4ECDC4')
    ax.fill(angles, debiased_values, alpha=0.25, color='#4ECDC4')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10)
    ax.set_ylim(0, 1)
    ax.set_title(f'Multi-Metric Comparison{title_suffix}', size=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_alpha_selection(results: List[Dict], output_path: Path, title_suffix: str = ""):
    """Plot alpha selection analysis."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    alphas = [r['alpha'] for r in results]
    orig_acc = [r['original_acc'] for r in results]
    deb_acc = [r['debiased_acc'] for r in results]
    improvements = [r['improvement'] for r in results]
    pos_vars = [r['pos_variance'] for r in results]
    
    best_idx = np.argmax(deb_acc)
    
    # Plot 1: Accuracies
    ax1.plot(alphas, orig_acc, 'o-', label='Original', linewidth=2, markersize=8, color='#FF6B6B')
    ax1.plot(alphas, deb_acc, 's-', label='Debiased', linewidth=2, markersize=8, color='#4ECDC4')
    ax1.axvline(x=alphas[best_idx], color='green', linestyle='--', linewidth=2, label=f'Best α={alphas[best_idx]:.2f}')
    ax1.set_xlabel('Alpha (Debiasing Strength)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Accuracy vs Alpha', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Improvement
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    ax2.bar(alphas, improvements, color=colors, alpha=0.7, width=0.08)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.axvline(x=alphas[best_idx], color='green', linestyle='--', linewidth=2, label=f'Best α={alphas[best_idx]:.2f}')
    ax2.set_xlabel('Alpha (Debiasing Strength)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy Improvement', fontsize=12, fontweight='bold')
    ax2.set_title('Accuracy Improvement vs Alpha', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Position Variance
    ax3.plot(alphas, pos_vars, 'D-', linewidth=2, markersize=8, color='#FFA07A')
    ax3.axvline(x=alphas[best_idx], color='green', linestyle='--', linewidth=2, label=f'Best α={alphas[best_idx]:.2f}')
    ax3.set_xlabel('Alpha (Debiasing Strength)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Position Variance', fontsize=12, fontweight='bold')
    ax3.set_title('Position Variance vs Alpha (lower = more uniform)', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Trade-off scatter
    ax4.scatter(pos_vars, deb_acc, c=alphas, cmap='viridis', s=200, alpha=0.7, edgecolors='black')
    ax4.scatter(pos_vars[best_idx], deb_acc[best_idx], c='red', s=400, marker='*', 
               edgecolors='black', linewidths=2, label=f'Best α={alphas[best_idx]:.2f}')
    for i, alpha in enumerate(alphas):
        ax4.annotate(f'{alpha:.1f}', (pos_vars[i], deb_acc[i]), fontsize=9, ha='center')
    ax4.set_xlabel('Position Variance (lower = more uniform)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Debiased Accuracy', fontsize=12, fontweight='bold')
    ax4.set_title('Accuracy vs Uniformity Trade-off', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    fig.suptitle(f'Alpha Selection Analysis{title_suffix}', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_summary_dashboard(baseline_metrics: Dict, debiased_metrics: Dict, best_alpha: float,
                           prior: List[float], output_path: Path, title_suffix: str = ""):
    """Create comprehensive summary dashboard."""
    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)
    
    # Title
    fig.suptitle(f'PriDe Comprehensive Analysis Dashboard{title_suffix}', 
                fontsize=20, fontweight='bold')
    
    # 1. Choice Distribution with Chi-square (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    positions = ['A', 'B', 'C', 'D']
    baseline_pcts = [baseline_metrics['choice_percentages'][p] for p in positions]
    debiased_pcts = [debiased_metrics['choice_percentages'][p] for p in positions]
    x = np.arange(len(positions))
    width = 0.35
    ax1.bar(x - width/2, baseline_pcts, width, label='Baseline', color='#FF6B6B', alpha=0.8)
    ax1.bar(x + width/2, debiased_pcts, width, label='After PriDe', color='#4ECDC4', alpha=0.8)
    ax1.axhline(y=25, color='red', linestyle='--', linewidth=2, label='Uniform (25%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(positions)
    ax1.set_ylabel('Percentage (%)', fontsize=10)
    ax1.set_title('Choice Distribution\n' + 
                  f'Baseline: χ²={baseline_metrics["chi2_stat"]:.2f}, p={baseline_metrics["chi2_pvalue"]:.4f}\n' +
                  f'PriDe: χ²={debiased_metrics["chi2_stat"]:.2f}, p={debiased_metrics["chi2_pvalue"]:.4f}',
                  fontweight='bold', fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Accuracy by Position with Chi-square (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    baseline_acc = [baseline_metrics['accuracy_by_position'].get(p, 0) for p in positions]
    debiased_acc = [debiased_metrics['accuracy_by_position'].get(p, 0) for p in positions]
    ax2.bar(x - width/2, baseline_acc, width, label='Baseline', color='#FF6B6B', alpha=0.8)
    ax2.bar(x + width/2, debiased_acc, width, label='After PriDe', color='#4ECDC4', alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(positions)
    ax2.set_ylabel('Accuracy', fontsize=10)
    ax2.set_title('Accuracy by Correct Position\n' +
                  f'Baseline: χ²={baseline_metrics["chi2_acc_stat"]:.2f}, p={baseline_metrics["chi2_acc_pvalue"]:.4f}\n' +
                  f'PriDe: χ²={debiased_metrics["chi2_acc_stat"]:.2f}, p={debiased_metrics["chi2_acc_pvalue"]:.4f}',
                  fontweight='bold', fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Estimated Prior (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    prior_pcts = np.array(prior) * 100
    bars = ax3.bar(positions, prior_pcts, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'], alpha=0.8)
    ax3.axhline(y=25, color='red', linestyle='--', linewidth=2, label='Uniform')
    ax3.set_ylabel('Probability (%)')
    ax3.set_title(f'Estimated Prior (α={best_alpha:.2f})', fontweight='bold')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 4. Key Metrics Table with Raw Values (middle section - spans 2 rows)
    ax4 = fig.add_subplot(gs[1:3, :2])
    ax4.axis('off')
    metrics_data = [
        ['Metric', 'Baseline', 'After PriDe', 'Change'],
        ['Overall Accuracy', f"{baseline_metrics['overall_accuracy']:.4f}", 
         f"{debiased_metrics['overall_accuracy']:.4f}",
         f"{debiased_metrics['overall_accuracy'] - baseline_metrics['overall_accuracy']:+.4f}"],
        ['Position Bias Score', f"{baseline_metrics['position_bias_score']:.2f}",
         f"{debiased_metrics['position_bias_score']:.2f}",
         f"{debiased_metrics['position_bias_score'] - baseline_metrics['position_bias_score']:+.2f}"],
        ['Recall Std (RStd) %', f"{baseline_metrics['recall_std']:.2f}",
         f"{debiased_metrics['recall_std']:.2f}",
         f"{debiased_metrics['recall_std'] - baseline_metrics['recall_std']:+.2f}"],
        ['', '', '', ''],  # Separator
        ['Distribution Chi-square (χ²)', f"{baseline_metrics['chi2_stat']:.2f}",
         f"{debiased_metrics['chi2_stat']:.2f}",
         f"{debiased_metrics['chi2_stat'] - baseline_metrics['chi2_stat']:+.2f}"],
        ['Distribution p-value', f"{baseline_metrics['chi2_pvalue']:.4f}",
         f"{debiased_metrics['chi2_pvalue']:.4f}",
         f"{debiased_metrics['chi2_pvalue'] - baseline_metrics['chi2_pvalue']:+.4f}"],
        ['', '', '', ''],  # Separator
        ['Acc-Position Chi-square (χ²)', f"{baseline_metrics['chi2_acc_stat']:.2f}",
         f"{debiased_metrics['chi2_acc_stat']:.2f}",
         f"{debiased_metrics['chi2_acc_stat'] - baseline_metrics['chi2_acc_stat']:+.2f}"],
        ['Acc-Position p-value', f"{baseline_metrics['chi2_acc_pvalue']:.4f}",
         f"{debiased_metrics['chi2_acc_pvalue']:.4f}",
         f"{debiased_metrics['chi2_acc_pvalue'] - baseline_metrics['chi2_acc_pvalue']:+.4f}"],
        ['', '', '', ''],  # Separator
        ['Choice A count', f"{baseline_metrics['choice_counts']['A']}", 
         f"{debiased_metrics['choice_counts']['A']}",
         f"{debiased_metrics['choice_counts']['A'] - baseline_metrics['choice_counts']['A']:+d}"],
        ['Choice B count', f"{baseline_metrics['choice_counts']['B']}", 
         f"{debiased_metrics['choice_counts']['B']}",
         f"{debiased_metrics['choice_counts']['B'] - baseline_metrics['choice_counts']['B']:+d}"],
        ['Choice C count', f"{baseline_metrics['choice_counts']['C']}", 
         f"{debiased_metrics['choice_counts']['C']}",
         f"{debiased_metrics['choice_counts']['C'] - baseline_metrics['choice_counts']['C']:+d}"],
        ['Choice D count', f"{baseline_metrics['choice_counts']['D']}", 
         f"{debiased_metrics['choice_counts']['D']}",
         f"{debiased_metrics['choice_counts']['D'] - baseline_metrics['choice_counts']['D']:+d}"],
    ]
    table = ax4.table(cellText=metrics_data, cellLoc='center', loc='center',
                     colWidths=[0.35, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style separator rows
    for i in [4, 7, 10]:
        if i < len(metrics_data):
            for j in range(4):
                table[(i, j)].set_facecolor('#E8E8E8')
    
    # Color code changes
    for i in range(1, len(metrics_data)):
        if metrics_data[i][0] == '':  # Skip separator rows
            continue
        change_text = metrics_data[i][3]
        if i == 1:  # Accuracy: positive is good
            if '+' in change_text and float(change_text) > 0:
                table[(i, 3)].set_facecolor('#90EE90')
            elif '-' in change_text:
                table[(i, 3)].set_facecolor('#FFB6C6')
        elif i in [2, 3]:  # Bias scores and Recall Std: negative is good (reduction)
            if '-' in change_text and float(change_text.replace('+', '').replace('-', '')) > 0:
                table[(i, 3)].set_facecolor('#90EE90')
            elif '+' in change_text:
                table[(i, 3)].set_facecolor('#FFB6C6')
        elif i in [5, 6, 8, 9]:  # Chi-square stats and p-values
            # For chi-square: lower is better (less deviation)
            # For p-values: higher is better (more uniform/independent)
            if i in [6, 9]:  # p-values
                if '+' in change_text:
                    table[(i, 3)].set_facecolor('#90EE90')
                elif '-' in change_text:
                    table[(i, 3)].set_facecolor('#FFB6C6')
            else:  # chi-square stats
                if '-' in change_text:
                    table[(i, 3)].set_facecolor('#90EE90')
                elif '+' in change_text:
                    table[(i, 3)].set_facecolor('#FFB6C6')
    
    ax4.set_title('Key Metrics Comparison (Raw Values + Statistics)', fontweight='bold', fontsize=12, pad=10)
    
    # 5. Statistical Summary Box (right side, spans rows 1-2)
    ax5 = fig.add_subplot(gs[1:3, 2])
    ax5.axis('off')
    acc_change = debiased_metrics['overall_accuracy'] - baseline_metrics['overall_accuracy']
    bias_change = debiased_metrics['position_bias_score'] - baseline_metrics['position_bias_score']
    chi2_change = debiased_metrics['chi2_pvalue'] - baseline_metrics['chi2_pvalue']
    rstd_change = debiased_metrics['recall_std'] - baseline_metrics['recall_std']
    
    interpretation = f"""INTERPRETATION & RAW VALUES
    
Alpha (α) = {best_alpha:.2f}

━━━ ACCURACY ━━━
Before: {baseline_metrics['overall_accuracy']:.4f}
After:  {debiased_metrics['overall_accuracy']:.4f}
Change: {acc_change:+.4f}
{'✅ Improved' if acc_change > 0 else '❌ Decreased'}

━━━ BIAS METRICS ━━━
Position Bias Score:
  Before: {baseline_metrics['position_bias_score']:.2f}
  After:  {debiased_metrics['position_bias_score']:.2f}
  Change: {bias_change:+.2f}
  {'✅ Reduced' if bias_change < 0 else '❌ Increased'}

Recall Std (RStd):
  Before: {baseline_metrics['recall_std']:.2f}%
  After:  {debiased_metrics['recall_std']:.2f}%
  Change: {rstd_change:+.2f}%
  {'✅ More balanced' if rstd_change < 0 else '❌ Less balanced'}

━━━ CHI-SQUARE TESTS ━━━
Distribution (χ², p-value):
  Before: χ²={baseline_metrics['chi2_stat']:.2f}, p={baseline_metrics['chi2_pvalue']:.4f}
  After:  χ²={debiased_metrics['chi2_stat']:.2f}, p={debiased_metrics['chi2_pvalue']:.4f}
  Δp: {chi2_change:+.4f}

Acc-Position (χ², p-value):
  Before: χ²={baseline_metrics['chi2_acc_stat']:.2f}, p={baseline_metrics['chi2_acc_pvalue']:.4f}
  After:  χ²={debiased_metrics['chi2_acc_stat']:.2f}, p={debiased_metrics['chi2_acc_pvalue']:.4f}

━━━ STRATEGY ━━━
"""
    
    if best_alpha < 0.3:
        interpretation += "⚠️ Minimal debiasing\n   Bias may be helpful"
    elif best_alpha < 0.7:
        interpretation += "⚖️ Partial debiasing\n   Balanced approach"
    else:
        interpretation += "✅ Strong debiasing\n   Bias is harmful"
    
    ax5.text(0.05, 0.95, interpretation, fontsize=8, verticalalignment='top',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            transform=ax5.transAxes)
    
    # 6. Position Changes Heatmap (bottom)
    ax6 = fig.add_subplot(gs[3, :])
    changes_matrix = []
    for pos in positions:
        baseline_val = baseline_metrics['accuracy_by_position'].get(pos, 0)
        debiased_val = debiased_metrics['accuracy_by_position'].get(pos, 0)
        change = debiased_val - baseline_val
        changes_matrix.append([baseline_val, debiased_val, change])
    
    changes_df = pd.DataFrame(changes_matrix, 
                              columns=['Baseline Accuracy', 'PriDe Accuracy', 'Change'],
                              index=positions)
    
    im = ax6.imshow(changes_df.T.values, cmap='RdYlGn', aspect='auto', vmin=-0.3, vmax=0.3)
    ax6.set_xticks(np.arange(len(positions)))
    ax6.set_yticks(np.arange(3))
    ax6.set_xticklabels(positions)
    ax6.set_yticklabels(changes_df.columns)
    ax6.set_title('Accuracy Heatmap by Correct Position', fontweight='bold', fontsize=14)
    
    # Add text annotations
    for i in range(3):
        for j in range(len(positions)):
            text = ax6.text(j, i, f'{changes_df.T.values[i, j]:.3f}',
                          ha="center", va="center", color="black", fontweight='bold')
    
    plt.colorbar(im, ax=ax6, label='Accuracy Value')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def test_alpha_values_with_viz(df: pd.DataFrame, calibration_ratio: float = 0.10,
                               viz_dir: Path = None):
    """Test different alpha values with visualization."""
    print("\n" + "=" * 70)
    print(f"TESTING ALPHA VALUES (Calibration={calibration_ratio:.0%})")
    print("=" * 70)
    
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results = []
    
    print("\nTesting alphas:", alphas)
    print("(This may take a minute...)\n")
    
    for alpha in alphas:
        pride = PriDeDebiasing(
            calibration_ratio=calibration_ratio,
            alpha=alpha,
            random_seed=42
        )
        
        test_df_debiased, calibration_info = pride.fit_and_predict(df)
        _, test_df_original = pride.split_calibration_test(df)
        
        test_df_original['is_correct_fixed'] = (
            test_df_original['predicted_answer'] == test_df_original['correct_position']
        ).astype(int)
        
        orig_acc = test_df_original['is_correct_fixed'].mean()
        deb_acc = test_df_debiased['debiased_is_correct'].mean()
        
        deb_dist = test_df_debiased['debiased_predicted_answer'].value_counts(normalize=True)
        pos_variance = deb_dist.var()
        
        results.append({
            'alpha': alpha,
            'original_acc': orig_acc,
            'debiased_acc': deb_acc,
            'improvement': deb_acc - orig_acc,
            'pos_variance': pos_variance
        })
    
    # Print results
    print(f"\n{'Alpha':<8} {'Original':>12} {'Debiased':>12} {'Improvement':>12} {'Pos Var':>12}")
    print("-" * 70)
    
    for r in results:
        marker = " 🏆" if r['debiased_acc'] == max(res['debiased_acc'] for res in results) else ""
        print(f"{r['alpha']:<8.2f} {r['original_acc']:>11.4f} {r['debiased_acc']:>11.4f} "
              f"{r['improvement']:>+11.4f} {r['pos_variance']:>11.6f}{marker}")
    
    best = max(results, key=lambda x: x['debiased_acc'])
    print(f"\n✅ Best alpha: {best['alpha']:.2f} "
          f"(Accuracy: {best['debiased_acc']:.4f}, Δ: {best['improvement']:+.4f})")
    
    return results, best['alpha']


def main_comprehensive_with_viz(csv_path: str, calibration_ratio: float = 0.10):
    """Comprehensive analysis with visualizations."""
    print("=" * 70)
    print("PriDe: COMPREHENSIVE VISUAL ANALYSIS")
    print("=" * 70)
    print(f"Dataset: {csv_path}")
    print(f"Calibration Ratio: {calibration_ratio:.0%}")
    print("=" * 70)
    
    dataset_name = Path(csv_path).stem
    
    # Load data
    print("\nLoading data...")
    df = load_and_prepare_data(csv_path)
    
    print(f"✅ Loaded successfully!")
    print(f"   Total questions: {df['question_id'].nunique()}")
    print(f"   Total samples: {len(df)}")
    
    if 'model' in df.columns and df['model'].notna().any():
        model_name_raw = str(df['model'].dropna().iloc[0])
    else:
        model_name_raw = "unknown_model"
    model_name = model_name_raw.replace(':', '_').replace('/', '_')
    
    # Create visualization folders
    base_dir, subdirs = create_visualization_folder(dataset_name, model_name)
    print(f"\n📁 Visualization folder: {base_dir}")
    
    # Test alpha values (this also determines the split)
    print("\n🔍 Testing alpha values to find optimal configuration...")
    results, best_alpha = test_alpha_values_with_viz(df, calibration_ratio=calibration_ratio, 
                                                     viz_dir=subdirs['alpha'])
    
    # Run PriDe with best alpha to get final results
    print(f"\n🔧 Running PriDe with optimal alpha={best_alpha:.2f}...")
    pride_final = PriDeDebiasing(
        calibration_ratio=calibration_ratio,
        alpha=best_alpha,
        random_seed=42
    )
    test_df_debiased_final, calibration_info_final = pride_final.fit_and_predict(df)
    
    # CRITICAL FIX: Get baseline from THE SAME test set that PriDe used
    _, test_df_original = pride_final.split_calibration_test(df)
    
    print(f"\n📊 Computing metrics on test set...")
    print(f"   Calibration: {calibration_info_final['n_calibration_questions']} questions")
    print(f"   Test: {calibration_info_final['n_test_questions']} questions")
    
    # Compute baseline metrics on TEST SET ONLY (not full dataset)
    baseline_metrics = compute_bias_metrics(test_df_original, 'predicted_answer')
    
    # Compute debiased metrics
    debiased_metrics = compute_bias_metrics(test_df_debiased_final, 'debiased_predicted_answer')
    
    # Print key results
    print(f"\n📈 KEY RESULTS (Test Set):")
    print(f"   Accuracy: {baseline_metrics['overall_accuracy']:.4f} → {debiased_metrics['overall_accuracy']:.4f} "
          f"({debiased_metrics['overall_accuracy'] - baseline_metrics['overall_accuracy']:+.4f})")
    print(f"   Position Bias Score: {baseline_metrics['position_bias_score']:.2f} → {debiased_metrics['position_bias_score']:.2f} "
          f"({debiased_metrics['position_bias_score'] - baseline_metrics['position_bias_score']:+.2f})")
    print(f"   Recall Std (RStd): {baseline_metrics['recall_std']:.2f}% → {debiased_metrics['recall_std']:.2f}% "
          f"({debiased_metrics['recall_std'] - baseline_metrics['recall_std']:+.2f}%)")
    
    # Generate all visualizations
    print("\n🎨 Generating visualizations...")
    
    print("  - Choice distribution comparison...")
    plot_choice_distribution_comparison(
        baseline_metrics, debiased_metrics,
        subdirs['distributions'] / 'choice_distribution_comparison.png',
        f" ({model_name})"
    )
    
    print("  - Accuracy by position comparison...")
    plot_accuracy_by_position_comparison(
        baseline_metrics, debiased_metrics,
        subdirs['accuracy'] / 'accuracy_by_position_comparison.png',
        f" ({model_name})"
    )
    
    print("  - Multi-metric radar chart...")
    plot_metrics_comparison_radar(
        baseline_metrics, debiased_metrics,
        subdirs['comparison'] / 'metrics_radar_comparison.png',
        f" ({model_name})"
    )
    
    print("  - Alpha selection analysis...")
    plot_alpha_selection(
        results,
        subdirs['alpha'] / 'alpha_selection_analysis.png',
        f" ({model_name})"
    )
    
    print("  - Summary dashboard...")
    plot_summary_dashboard(
        baseline_metrics, debiased_metrics, best_alpha,
        calibration_info_final['estimated_prior'],
        base_dir / 'SUMMARY_DASHBOARD.png',
        f" ({model_name})"
    )
    
    # Save numerical results
    print("\n💾 Saving numerical results...")
    csv_output_dir = Path("pride/results/csv")
    csv_output_dir.mkdir(parents=True, exist_ok=True)
    output_csv_path = csv_output_dir / f"{dataset_name}_pride_debiased.csv"
    test_df_debiased_final.to_csv(output_csv_path, index=False)
    
    # Save comprehensive text report with all raw values and p-values
    report_path = base_dir / 'analysis_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("PriDe COMPREHENSIVE DEBIASING ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Dataset: {csv_path}\n")
        f.write(f"Model: {model_name}\n")
        f.write("=" * 80 + "\n\n")
        
        # Configuration
        f.write("CONFIGURATION:\n")
        f.write(f"  Calibration Ratio: {calibration_ratio:.0%}\n")
        f.write(f"  Alpha (Debiasing Strength): {best_alpha:.2f}\n")
        f.write(f"  Calibration Questions: {calibration_info_final['n_calibration_questions']}\n")
        f.write(f"  Test Questions: {calibration_info_final['n_test_questions']}\n")
        f.write(f"  Total Evaluations (Test Set): {baseline_metrics['n_samples']}\n\n")
        
        # ========================================================================
        # 1. CHOICE DISTRIBUTION
        # ========================================================================
        f.write("=" * 80 + "\n")
        f.write("1. CHOICE DISTRIBUTION (Predicted Answers)\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("BASELINE (Before PriDe):\n")
        f.write(f"{'Position':<12} {'Count':<10} {'Percentage':<15} {'Expected':<15} {'Deviation':<15}\n")
        f.write("-" * 80 + "\n")
        total = baseline_metrics['n_samples']
        expected = total / 4
        for pos in ['A', 'B', 'C', 'D']:
            count = baseline_metrics['choice_counts'][pos]
            pct = baseline_metrics['choice_percentages'][pos]
            dev = count - expected
            f.write(f"{pos:<12} {count:<10} {pct:>6.2f}% {' '*8} {expected:>6.1f} {' '*8} {dev:>+6.1f}\n")
        
        f.write("\nAFTER PriDe:\n")
        f.write(f"{'Position':<12} {'Count':<10} {'Percentage':<15} {'Expected':<15} {'Deviation':<15}\n")
        f.write("-" * 80 + "\n")
        for pos in ['A', 'B', 'C', 'D']:
            count = debiased_metrics['choice_counts'][pos]
            pct = debiased_metrics['choice_percentages'][pos]
            dev = count - expected
            f.write(f"{pos:<12} {count:<10} {pct:>6.2f}% {' '*8} {expected:>6.1f} {' '*8} {dev:>+6.1f}\n")
        
        # ========================================================================
        # 2. CHI-SQUARE TEST: Distribution vs Uniform
        # ========================================================================
        f.write("\n" + "=" * 80 + "\n")
        f.write("2. CHI-SQUARE TEST: Choice Distribution vs Uniform (25/25/25/25)\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("BASELINE:\n")
        f.write(f"  Observed Frequencies: A={baseline_metrics['choice_counts']['A']}, "
                f"B={baseline_metrics['choice_counts']['B']}, "
                f"C={baseline_metrics['choice_counts']['C']}, "
                f"D={baseline_metrics['choice_counts']['D']}\n")
        f.write(f"  Expected Frequencies: A={expected:.1f}, B={expected:.1f}, "
                f"C={expected:.1f}, D={expected:.1f}\n")
        f.write(f"  Chi-square Statistic (χ²): {baseline_metrics['chi2_stat']:.4f}\n")
        f.write(f"  Degrees of Freedom: 3\n")
        f.write(f"  P-value: {baseline_metrics['chi2_pvalue']:.6f}\n")
        if baseline_metrics['chi2_pvalue'] < 0.001:
            f.write(f"  Interpretation: *** Highly significant deviation (p < 0.001)\n")
        elif baseline_metrics['chi2_pvalue'] < 0.01:
            f.write(f"  Interpretation: ** Significant deviation (p < 0.01)\n")
        elif baseline_metrics['chi2_pvalue'] < 0.05:
            f.write(f"  Interpretation: * Significant deviation (p < 0.05)\n")
        else:
            f.write(f"  Interpretation: No significant deviation (p >= 0.05)\n")
        
        f.write("\nAFTER PriDe:\n")
        f.write(f"  Observed Frequencies: A={debiased_metrics['choice_counts']['A']}, "
                f"B={debiased_metrics['choice_counts']['B']}, "
                f"C={debiased_metrics['choice_counts']['C']}, "
                f"D={debiased_metrics['choice_counts']['D']}\n")
        f.write(f"  Expected Frequencies: A={expected:.1f}, B={expected:.1f}, "
                f"C={expected:.1f}, D={expected:.1f}\n")
        f.write(f"  Chi-square Statistic (χ²): {debiased_metrics['chi2_stat']:.4f}\n")
        f.write(f"  Degrees of Freedom: 3\n")
        f.write(f"  P-value: {debiased_metrics['chi2_pvalue']:.6f}\n")
        if debiased_metrics['chi2_pvalue'] < 0.001:
            f.write(f"  Interpretation: *** Highly significant deviation (p < 0.001)\n")
        elif debiased_metrics['chi2_pvalue'] < 0.01:
            f.write(f"  Interpretation: ** Significant deviation (p < 0.01)\n")
        elif debiased_metrics['chi2_pvalue'] < 0.05:
            f.write(f"  Interpretation: * Significant deviation (p < 0.05)\n")
        else:
            f.write(f"  Interpretation: No significant deviation (p >= 0.05)\n")
        
        f.write(f"\nCHANGE:\n")
        f.write(f"  Δχ²: {debiased_metrics['chi2_stat'] - baseline_metrics['chi2_stat']:+.4f}\n")
        f.write(f"  Δp-value: {debiased_metrics['chi2_pvalue'] - baseline_metrics['chi2_pvalue']:+.6f}\n")
        
        # ========================================================================
        # 3. POSITION BIAS SCORE
        # ========================================================================
        f.write("\n" + "=" * 80 + "\n")
        f.write("3. POSITION BIAS SCORE (Standard Deviation of Choice Percentages)\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("BASELINE:\n")
        f.write(f"  Choice Percentages: A={baseline_metrics['choice_percentages']['A']:.2f}%, "
                f"B={baseline_metrics['choice_percentages']['B']:.2f}%, "
                f"C={baseline_metrics['choice_percentages']['C']:.2f}%, "
                f"D={baseline_metrics['choice_percentages']['D']:.2f}%\n")
        f.write(f"  Position Bias Score: {baseline_metrics['position_bias_score']:.4f}\n")
        
        f.write("\nAFTER PriDe:\n")
        f.write(f"  Choice Percentages: A={debiased_metrics['choice_percentages']['A']:.2f}%, "
                f"B={debiased_metrics['choice_percentages']['B']:.2f}%, "
                f"C={debiased_metrics['choice_percentages']['C']:.2f}%, "
                f"D={debiased_metrics['choice_percentages']['D']:.2f}%\n")
        f.write(f"  Position Bias Score: {debiased_metrics['position_bias_score']:.4f}\n")
        
        f.write(f"\nCHANGE: {debiased_metrics['position_bias_score'] - baseline_metrics['position_bias_score']:+.4f}")
        if debiased_metrics['position_bias_score'] < baseline_metrics['position_bias_score']:
            f.write(" ✅ (Improved - bias reduced)\n")
        else:
            f.write(" ⚠️ (Bias increased)\n")
        
        # ========================================================================
        # 4. ACCURACY ANALYSIS
        # ========================================================================
        f.write("\n" + "=" * 80 + "\n")
        f.write("4. ACCURACY BY CORRECT ANSWER POSITION\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("BASELINE:\n")
        f.write(f"  Overall Accuracy: {baseline_metrics['overall_accuracy']:.4f} ({baseline_metrics['overall_accuracy']*100:.2f}%)\n")
        f.write(f"\n  {'Position':<12} {'Accuracy':<15} {'Deviation from Mean':<20}\n")
        f.write("  " + "-" * 50 + "\n")
        for pos in ['A', 'B', 'C', 'D']:
            acc = baseline_metrics['accuracy_by_position'].get(pos, 0)
            dev = acc - baseline_metrics['overall_accuracy']
            f.write(f"  {pos:<12} {acc:<15.4f} {dev:>+7.4f}\n")
        
        f.write("\nAFTER PriDe:\n")
        f.write(f"  Overall Accuracy: {debiased_metrics['overall_accuracy']:.4f} ({debiased_metrics['overall_accuracy']*100:.2f}%)\n")
        f.write(f"\n  {'Position':<12} {'Accuracy':<15} {'Deviation from Mean':<20}\n")
        f.write("  " + "-" * 50 + "\n")
        for pos in ['A', 'B', 'C', 'D']:
            acc = debiased_metrics['accuracy_by_position'].get(pos, 0)
            dev = acc - debiased_metrics['overall_accuracy']
            f.write(f"  {pos:<12} {acc:<15.4f} {dev:>+7.4f}\n")
        
        f.write(f"\nOVERALL ACCURACY CHANGE: {debiased_metrics['overall_accuracy'] - baseline_metrics['overall_accuracy']:+.4f}")
        if debiased_metrics['overall_accuracy'] > baseline_metrics['overall_accuracy']:
            f.write(" ✅ (Improved)\n")
        else:
            f.write(" ⚠️ (Decreased)\n")
        
        # ========================================================================
        # 5. CHI-SQUARE TEST: Accuracy vs Position Independence
        # ========================================================================
        f.write("\n" + "=" * 80 + "\n")
        f.write("5. CHI-SQUARE TEST: Accuracy vs Position (Independence Test)\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("BASELINE:\n")
        f.write(f"  Chi-square Statistic (χ²): {baseline_metrics['chi2_acc_stat']:.4f}\n")
        f.write(f"  P-value: {baseline_metrics['chi2_acc_pvalue']:.6f}\n")
        if baseline_metrics['chi2_acc_pvalue'] < 0.001:
            f.write(f"  Interpretation: *** Accuracy STRONGLY depends on position (p < 0.001)\n")
        elif baseline_metrics['chi2_acc_pvalue'] < 0.01:
            f.write(f"  Interpretation: ** Accuracy depends on position (p < 0.01)\n")
        elif baseline_metrics['chi2_acc_pvalue'] < 0.05:
            f.write(f"  Interpretation: * Accuracy may depend on position (p < 0.05)\n")
        else:
            f.write(f"  Interpretation: Accuracy independent of position (p >= 0.05)\n")
        
        f.write("\nAFTER PriDe:\n")
        f.write(f"  Chi-square Statistic (χ²): {debiased_metrics['chi2_acc_stat']:.4f}\n")
        f.write(f"  P-value: {debiased_metrics['chi2_acc_pvalue']:.6f}\n")
        if debiased_metrics['chi2_acc_pvalue'] < 0.001:
            f.write(f"  Interpretation: *** Accuracy STRONGLY depends on position (p < 0.001)\n")
        elif debiased_metrics['chi2_acc_pvalue'] < 0.01:
            f.write(f"  Interpretation: ** Accuracy depends on position (p < 0.01)\n")
        elif debiased_metrics['chi2_acc_pvalue'] < 0.05:
            f.write(f"  Interpretation: * Accuracy may depend on position (p < 0.05)\n")
        else:
            f.write(f"  Interpretation: Accuracy independent of position (p >= 0.05)\n")
        
        f.write(f"\nCHANGE:\n")
        f.write(f"  Δχ²: {debiased_metrics['chi2_acc_stat'] - baseline_metrics['chi2_acc_stat']:+.4f}\n")
        f.write(f"  Δp-value: {debiased_metrics['chi2_acc_pvalue'] - baseline_metrics['chi2_acc_pvalue']:+.6f}")
        if debiased_metrics['chi2_acc_pvalue'] > baseline_metrics['chi2_acc_pvalue']:
            f.write(" ✅ (More independent)\n")
        else:
            f.write(" ⚠️ (More dependent)\n")
        
        # ========================================================================
        # 6. RECALL ANALYSIS (PriDe Paper Metric)
        # ========================================================================
        f.write("\n" + "=" * 80 + "\n")
        f.write("6. RECALL BY POSITION (PriDe Paper's RStd Metric)\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("BASELINE:\n")
        f.write(f"  {'Position':<12} {'Recall Rate':<15}\n")
        f.write("  " + "-" * 30 + "\n")
        for pos in ['A', 'B', 'C', 'D']:
            recall = baseline_metrics['recalls'][pos]
            f.write(f"  {pos:<12} {recall:<15.4f} ({recall*100:.2f}%)\n")
        f.write(f"  Recall Std Dev (RStd): {baseline_metrics['recall_std']:.2f}%\n")
        
        f.write("\nAFTER PriDe:\n")
        f.write(f"  {'Position':<12} {'Recall Rate':<15}\n")
        f.write("  " + "-" * 30 + "\n")
        for pos in ['A', 'B', 'C', 'D']:
            recall = debiased_metrics['recalls'][pos]
            f.write(f"  {pos:<12} {recall:<15.4f} ({recall*100:.2f}%)\n")
        f.write(f"  Recall Std Dev (RStd): {debiased_metrics['recall_std']:.2f}%\n")
        
        f.write(f"\nCHANGE: {debiased_metrics['recall_std'] - baseline_metrics['recall_std']:+.2f}%")
        if debiased_metrics['recall_std'] < baseline_metrics['recall_std']:
            f.write(" ✅ (Improved - more balanced)\n")
        else:
            f.write(" ⚠️ (More imbalanced)\n")
        
        # ========================================================================
        # 7. ESTIMATED PRIOR (TOKEN BIAS)
        # ========================================================================
        f.write("\n" + "=" * 80 + "\n")
        f.write("7. ESTIMATED PRIOR (Token Bias from Calibration Set)\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"{'Position':<12} {'Prior Probability':<20} {'Percentage':<15}\n")
        f.write("-" * 50 + "\n")
        for i, pos in enumerate(['A', 'B', 'C', 'D']):
            prior_prob = calibration_info_final['estimated_prior'][i]
            f.write(f"{pos:<12} {prior_prob:<20.4f} {prior_prob*100:>6.2f}%\n")
        f.write(f"\nNote: Prior represents the model's inherent bias toward each token/position.\n")
        f.write(f"      Uniform (unbiased) would be: A=0.25, B=0.25, C=0.25, D=0.25\n")
        
        # ========================================================================
        # 8. SUMMARY TABLE
        # ========================================================================
        f.write("\n" + "=" * 80 + "\n")
        f.write("8. SUMMARY TABLE\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"{'Metric':<35} {'Baseline':<15} {'After PriDe':<15} {'Change':<15}\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Overall Accuracy':<35} {baseline_metrics['overall_accuracy']:<15.4f} "
                f"{debiased_metrics['overall_accuracy']:<15.4f} "
                f"{debiased_metrics['overall_accuracy'] - baseline_metrics['overall_accuracy']:<+15.4f}\n")
        f.write(f"{'Position Bias Score':<35} {baseline_metrics['position_bias_score']:<15.4f} "
                f"{debiased_metrics['position_bias_score']:<15.4f} "
                f"{debiased_metrics['position_bias_score'] - baseline_metrics['position_bias_score']:<+15.4f}\n")
        f.write(f"{'Recall Std (RStd) %':<35} {baseline_metrics['recall_std']:<15.2f} "
                f"{debiased_metrics['recall_std']:<15.2f} "
                f"{debiased_metrics['recall_std'] - baseline_metrics['recall_std']:<+15.2f}\n")
        f.write(f"{'Chi² (Distribution)':<35} {baseline_metrics['chi2_stat']:<15.4f} "
                f"{debiased_metrics['chi2_stat']:<15.4f} "
                f"{debiased_metrics['chi2_stat'] - baseline_metrics['chi2_stat']:<+15.4f}\n")
        f.write(f"{'Chi² p-value (Distribution)':<35} {baseline_metrics['chi2_pvalue']:<15.6f} "
                f"{debiased_metrics['chi2_pvalue']:<15.6f} "
                f"{debiased_metrics['chi2_pvalue'] - baseline_metrics['chi2_pvalue']:<+15.6f}\n")
        f.write(f"{'Chi² (Accuracy vs Position)':<35} {baseline_metrics['chi2_acc_stat']:<15.4f} "
                f"{debiased_metrics['chi2_acc_stat']:<15.4f} "
                f"{debiased_metrics['chi2_acc_stat'] - baseline_metrics['chi2_acc_stat']:<+15.4f}\n")
        f.write(f"{'Chi² p-value (Acc vs Pos)':<35} {baseline_metrics['chi2_acc_pvalue']:<15.6f} "
                f"{debiased_metrics['chi2_acc_pvalue']:<15.6f} "
                f"{debiased_metrics['chi2_acc_pvalue'] - baseline_metrics['chi2_acc_pvalue']:<+15.6f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"✅ Report saved to: {report_path}")
    print(f"✅ CSV saved to: {output_csv_path}")
    
    print("\n" + "=" * 70)
    print("✨ ANALYSIS COMPLETE! ✨")
    print("=" * 70)
    print(f"\n📂 All visualizations saved to: {base_dir}")
    print(f"   - Summary Dashboard: SUMMARY_DASHBOARD.png")
    print(f"   - Subdirectories organized by analysis type")
    
    return results, best_alpha, base_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PriDe debiasing with comprehensive visualizations.")
    parser.add_argument("csv_path", help="Path to the input CSV with model probabilities.")
    parser.add_argument("--calibration-ratio", type=float, default=0.15,
                       help="Ratio of data to use for calibration (default: 0.10 = 10%%)")
    args = parser.parse_args()
    results, best_alpha, viz_dir = main_comprehensive_with_viz(args.csv_path, args.calibration_ratio)