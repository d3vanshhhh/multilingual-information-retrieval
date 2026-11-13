#!/usr/bin/env python
"""
Compare v1 (old nllb) vs v2 (new nllb-v2) translation performance.
Reads per-query nDCG@10 results and performs paired bootstrap significance testing.
"""
import yaml
import pandas as pd
import numpy as np
from pathlib import Path


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def bootstrap_ci_paired(differences, n_resamples=2000, confidence=0.95, seed=42):
    """Compute bootstrap confidence interval for mean of paired differences"""
    rng = np.random.RandomState(seed)
    bootstrap_means = []
    
    for _ in range(n_resamples):
        resample = rng.choice(differences, size=len(differences), replace=True)
        bootstrap_means.append(np.mean(resample))
    
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = np.percentile(bootstrap_means, lower_percentile)
    ci_upper = np.percentile(bootstrap_means, upper_percentile)
    
    return ci_lower, ci_upper


def main():
    config = load_config()
    models = list(config['models'].keys())
    languages = ['de', 'hi']
    
    results_dir = Path('results/per_query')
    v1_backup_dir = Path('logs/v1_backup/per_query')
    
    print("="*120)
    print("V1 vs V2 TRANSLATION COMPARISON")
    print("="*120)
    
    all_results = []
    
    for model in models:
        for lang in languages:
            print(f"\nProcessing {model} / {lang}...")
            
            # Load direct results
            direct_file = results_dir / f"{model}_direct_{lang}.csv"
            if not direct_file.exists():
                print(f"  WARNING: Direct file not found: {direct_file}")
                continue
            direct_df = pd.read_csv(direct_file)
            
            # Load v2 translate results
            v2_file = results_dir / f"{model}_translate_{lang}.csv"
            if not v2_file.exists():
                print(f"  WARNING: V2 translate file not found: {v2_file}")
                continue
            v2_df = pd.read_csv(v2_file)
            
            # Load v1 translate results
            v1_file = v1_backup_dir / f"{model}_translate_{lang}.csv"
            if not v1_file.exists():
                print(f"  WARNING: V1 translate file not found: {v1_file}")
                continue
            v1_df = pd.read_csv(v1_file)
            
            # Merge by query_id
            merged = direct_df[['query_id', 'ndcg@10']].rename(columns={'ndcg@10': 'direct'})
            merged = merged.merge(
                v1_df[['query_id', 'ndcg@10']].rename(columns={'ndcg@10': 'translate_v1'}),
                on='query_id',
                how='inner'
            )
            merged = merged.merge(
                v2_df[['query_id', 'ndcg@10']].rename(columns={'ndcg@10': 'translate_v2'}),
                on='query_id',
                how='inner'
            )
            
            # Check if query_id sets match
            direct_ids = set(direct_df['query_id'])
            v1_ids = set(v1_df['query_id'])
            v2_ids = set(v2_df['query_id'])
            
            if direct_ids != v1_ids or direct_ids != v2_ids:
                raise ValueError(f"Query ID sets differ for {model}/{lang}: "
                               f"direct={len(direct_ids)}, v1={len(v1_ids)}, v2={len(v2_ids)}")
            
            print(f"  Merged {len(merged)} queries")
            
            # Compute means
            direct_mean = merged['direct'].mean()
            v1_mean = merged['translate_v1'].mean()
            v2_mean = merged['translate_v2'].mean()
            
            # Compute paired difference and bootstrap CI
            differences = (merged['translate_v2'] - merged['translate_v1']).values
            ci_lower, ci_upper = bootstrap_ci_paired(differences, n_resamples=2000, seed=42)
            
            # Determine winner
            winner = 'Direct' if direct_mean > v2_mean else 'Translate'
            
            result = {
                'Model': model,
                'Language': lang,
                'Direct': direct_mean,
                'Translate_v1': v1_mean,
                'Translate_v2': v2_mean,
                'v2-v1': v2_mean - v1_mean,
                'CI95_low': ci_lower,
                'CI95_high': ci_upper,
                'Winner_v2': winner
            }
            
            all_results.append(result)
            
            print(f"    Direct: {direct_mean:.4f}")
            print(f"    Translate v1: {v1_mean:.4f}")
            print(f"    Translate v2: {v2_mean:.4f}")
            print(f"    Difference (v2-v1): {v2_mean - v1_mean:.4f}")
            print(f"    95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
            print(f"    Winner: {winner}")
    
    # Create DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save to CSV
    output_file = Path('results/v1_vs_v2.csv')
    results_df.to_csv(output_file, index=False)
    print(f"\nSaved results to {output_file}")
    
    # Print report
    print("\n" + "="*120)
    print("=== REPORT ===")
    print("="*120)
    
    print("\nV1 vs V2 Translation Comparison (nDCG@10):")
    print(f"{'Model':<30} {'Lang':<4} {'Direct':<8} {'Trans_v1':<10} {'Trans_v2':<10} "
          f"{'v2-v1':<8} {'CI95_low':<9} {'CI95_high':<9} {'Winner_v2':<10}")
    print("-"*120)
    
    for _, row in results_df.iterrows():
        print(f"{row['Model']:<30} {row['Language']:<4} {row['Direct']:<8.4f} "
              f"{row['Translate_v1']:<10.4f} {row['Translate_v2']:<10.4f} "
              f"{row['v2-v1']:<8.4f} {row['CI95_low']:<9.4f} {row['CI95_high']:<9.4f} "
              f"{row['Winner_v2']:<10}")
    
    print("="*120)


if __name__ == '__main__':
    main()
