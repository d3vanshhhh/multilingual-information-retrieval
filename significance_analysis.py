"""
Statistical significance analysis comparing direct vs translate pipelines.

Implements paired randomization test (sign-flip permutation test) following:
Smucker, Allan & Carterette. "A comparison of statistical significance tests
for information retrieval evaluation." CIKM 2007.

For each (model, language) pair:
- Computes mean difference in nDCG@10 (translate - direct)
- 95% CI via paired bootstrap (percentile method)
- Two-sided p-value via paired randomization test (sign-flip permutations)
- Holm-Bonferroni correction for multiple comparisons
"""
import numpy as np
import pandas as pd
import yaml
from pathlib import Path


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def paired_bootstrap_ci(differences, n_resamples=10000, alpha=0.05, rng=None):
    """
    Compute confidence interval for mean of paired differences via bootstrap.
    
    Args:
        differences: Array of paired differences
        n_resamples: Number of bootstrap resamples
        alpha: Significance level (default 0.05 for 95% CI)
        rng: NumPy random generator
    
    Returns:
        (ci_low, ci_high): 95% confidence interval
    """
    n = len(differences)
    bootstrap_means = []
    
    # Process in chunks to keep memory usage low
    chunk_size = 1000
    for chunk_start in range(0, n_resamples, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_resamples)
        chunk_resamples = chunk_end - chunk_start
        
        # Resample with replacement
        for _ in range(chunk_resamples):
            resample = rng.choice(differences, size=n, replace=True)
            bootstrap_means.append(np.mean(resample))
    
    # Percentile method
    ci_low = np.percentile(bootstrap_means, 100 * alpha / 2)
    ci_high = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    
    return ci_low, ci_high


def paired_randomization_test(differences, n_permutations=10000, rng=None):
    """
    Two-sided paired randomization test (sign-flip permutation test).
    
    Args:
        differences: Array of paired differences
        n_permutations: Number of random permutations
        rng: NumPy random generator
    
    Returns:
        p_value: Two-sided p-value
    """
    observed_mean = np.mean(differences)
    observed_abs_mean = np.abs(observed_mean)
    
    # Count permutations where |mean| >= |observed mean|
    extreme_count = 1  # Start with 1 for the observed data itself
    
    # Process in chunks to keep memory usage low
    chunk_size = 1000
    for chunk_start in range(0, n_permutations, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_permutations)
        chunk_permutations = chunk_end - chunk_start
        
        # Random sign flips
        for _ in range(chunk_permutations):
            signs = rng.choice([-1, 1], size=len(differences))
            permuted = differences * signs
            permuted_mean = np.mean(permuted)
            
            if np.abs(permuted_mean) >= observed_abs_mean:
                extreme_count += 1
    
    # Two-sided p-value with continuity correction
    p_value = extreme_count / (n_permutations + 1)
    
    return p_value


def holm_bonferroni_correction(p_values):
    """
    Apply Holm-Bonferroni correction for multiple comparisons.
    
    Args:
        p_values: Array of p-values
    
    Returns:
        adjusted_p_values: Holm-corrected p-values
    """
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]
    
    # Holm correction: p_adj = max(p * (n - rank + 1))
    adjusted = np.zeros(n)
    for i, p in enumerate(sorted_p):
        adjusted[sorted_indices[i]] = p * (n - i)
    
    # Enforce monotonicity (adjusted p-values must be non-decreasing)
    adjusted = np.maximum.accumulate(adjusted[sorted_indices])[np.argsort(sorted_indices)]
    
    # Cap at 1.0
    adjusted = np.minimum(adjusted, 1.0)
    
    return adjusted


def verify_against_winloss(results_df, winloss_path='results/winloss.csv'):
    """
    Verify that Mean_Diff matches Margin in winloss.csv for consistency.
    
    Raises:
        AssertionError: If any row differs by more than 1e-9
    """
    winloss_df = pd.read_csv(winloss_path)
    
    for _, row in results_df.iterrows():
        model = row['Model']
        language = row['Language']
        mean_diff = row['Mean_Diff']
        
        # Find corresponding winloss row
        winloss_row = winloss_df[
            (winloss_df['Model'] == model) & 
            (winloss_df['Language'] == language)
        ]
        
        if winloss_row.empty:
            raise AssertionError(
                f"No winloss entry found for Model={model}, Language={language}"
            )
        
        margin = winloss_row['Margin'].iloc[0]
        diff = abs(mean_diff - margin)
        
        if diff > 1e-9:
            raise AssertionError(
                f"Consistency check failed for Model={model}, Language={language}: "
                f"Mean_Diff={mean_diff}, Margin={margin}, diff={diff}"
            )


def main():
    print("="*80)
    print("SIGNIFICANCE ANALYSIS: Direct vs Translate Pipelines")
    print("="*80)
    print("Method: Paired randomization test (sign-flip permutations)")
    print("Reference: Smucker, Allan & Carterette, CIKM 2007")
    print("="*80)
    print()
    
    # Load config
    config = load_config()
    models = list(config['models'].keys())
    languages = ['de', 'hi']  # Only non-English languages have translate pipeline
    
    # Initialize random generator with fixed seed for reproducibility
    rng = np.random.default_rng(42)
    
    results = []
    
    for model in models:
        for language in languages:
            print(f"Processing: {model} / {language}")
            
            # Load per-query metrics
            direct_path = Path('results') / 'per_query' / f"{model}_direct_{language}.csv"
            translate_path = Path('results') / 'per_query' / f"{model}_translate_{language}.csv"
            
            if not direct_path.exists():
                raise FileNotFoundError(f"Direct pipeline results not found: {direct_path}")
            if not translate_path.exists():
                raise FileNotFoundError(f"Translate pipeline results not found: {translate_path}")
            
            direct_df = pd.read_csv(direct_path)
            translate_df = pd.read_csv(translate_path)
            
            # Inner join on query_id
            merged_df = pd.merge(
                direct_df, translate_df,
                on='query_id',
                suffixes=('_direct', '_translate')
            )
            
            # Assert same query IDs (inner join should match if both have same queries)
            if len(merged_df) != 1500:
                raise AssertionError(
                    f"Expected 1500 queries for {model}/{language}, "
                    f"got {len(merged_df)} after inner join. "
                    f"Direct has {len(direct_df)} queries, translate has {len(translate_df)}."
                )
            
            # Ensure query_ids match exactly
            direct_ids = set(direct_df['query_id'])
            translate_ids = set(translate_df['query_id'])
            if direct_ids != translate_ids:
                missing_in_translate = direct_ids - translate_ids
                missing_in_direct = translate_ids - direct_ids
                raise AssertionError(
                    f"Query ID mismatch for {model}/{language}. "
                    f"Missing in translate: {len(missing_in_translate)}, "
                    f"missing in direct: {len(missing_in_direct)}"
                )
            
            # Compute per-query differences (translate - direct)
            differences = (merged_df['ndcg@10_translate'] - merged_df['ndcg@10_direct']).values
            
            # Mean difference
            mean_diff = np.mean(differences)
            
            # Mean scores
            direct_mean = np.mean(merged_df['ndcg@10_direct'])
            translate_mean = np.mean(merged_df['ndcg@10_translate'])
            
            # 95% CI via paired bootstrap
            print(f"  Computing 95% CI via bootstrap (10,000 resamples)...")
            ci_low, ci_high = paired_bootstrap_ci(differences, n_resamples=10000, rng=rng)
            
            # Two-sided p-value via paired randomization test
            print(f"  Computing p-value via randomization test (10,000 permutations)...")
            p_value = paired_randomization_test(differences, n_permutations=10000, rng=rng)
            
            # Winner
            winner = 'Translate' if mean_diff > 0 else 'Direct'
            
            results.append({
                'Model': model,
                'Language': language,
                'Direct_nDCG@10': direct_mean,
                'Translate_nDCG@10': translate_mean,
                'Mean_Diff': mean_diff,
                'CI95_Low': ci_low,
                'CI95_High': ci_high,
                'p_value': p_value,
                'Winner': winner
            })
            
            print(f"  Mean diff: {mean_diff:.4f}, p-value: {p_value:.4e}")
            print()
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Holm-Bonferroni correction
    print("Applying Holm-Bonferroni correction for multiple comparisons...")
    results_df['p_holm'] = holm_bonferroni_correction(results_df['p_value'].values)
    results_df['Significant'] = results_df['p_holm'] < 0.05
    
    # Verify consistency with winloss.csv
    print("Verifying consistency with results/winloss.csv...")
    verify_against_winloss(results_df)
    print("✓ Consistency check passed!")
    print()
    
    # Sort by Model then Language
    results_df = results_df.sort_values(['Model', 'Language'])
    
    # Save to CSV
    output_path = Path('results') / 'significance.csv'
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    print()
    
    # Print full table to console
    print("="*80)
    print("RESULTS")
    print("="*80)
    pd.options.display.float_format = '{:.4f}'.format
    
    # Format for display (scientific notation for p-values)
    display_df = results_df.copy()
    display_df['p_value'] = display_df['p_value'].apply(lambda x: f"{x:.4e}")
    display_df['p_holm'] = display_df['p_holm'].apply(lambda x: f"{x:.4e}")
    
    print(display_df.to_string(index=False))
    print()
    print("="*80)
    
    # Summary statistics
    n_significant = results_df['Significant'].sum()
    n_total = len(results_df)
    print(f"Significant differences: {n_significant}/{n_total} ({100*n_significant/n_total:.1f}%)")
    print("="*80)


if __name__ == "__main__":
    main()
