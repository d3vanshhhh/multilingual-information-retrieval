"""
Quick script to compare results and generate visualizations.
Run after evaluation is complete.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import argparse


def load_results():
    """Load evaluation results."""
    summary_path = Path('results/summary.csv')
    winloss_path = Path('results/winloss.csv')
    
    if not summary_path.exists():
        print("❌ No results found. Run evaluate.py first.")
        return None, None
    
    summary = pd.read_csv(summary_path)
    winloss = pd.read_csv(winloss_path) if winloss_path.exists() else None
    
    return summary, winloss


def plot_model_comparison(summary_df):
    """Plot nDCG@10 comparison across models."""
    if 'avg_ndcg@10' not in summary_df.columns:
        print("⚠️  nDCG@10 column not found")
        return
    
    fig = px.bar(
        summary_df,
        x='model',
        y='avg_ndcg@10',
        color='pipeline',
        barmode='group',
        title='Model Performance Comparison (nDCG@10)',
        labels={'avg_ndcg@10': 'Average nDCG@10', 'model': 'Model'},
        color_discrete_map={'direct': '#1f77b4', 'translate': '#ff7f0e'}
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        height=500,
        showlegend=True
    )
    
    fig.write_html('results/model_comparison.html')
    print("✅ Saved: results/model_comparison.html")
    fig.show()


def plot_pipeline_winloss(winloss_df):
    """Plot win/loss analysis."""
    if winloss_df is None:
        print("⚠️  No win/loss data found")
        return
    
    # Count wins
    win_counts = winloss_df['Winner'].value_counts()
    
    fig = px.pie(
        values=win_counts.values,
        names=win_counts.index,
        title='Pipeline Win/Loss Across Languages',
        color=win_counts.index,
        color_discrete_map={'Direct': '#1f77b4', 'Translate': '#ff7f0e'}
    )
    
    fig.write_html('results/pipeline_winloss.html')
    print("✅ Saved: results/pipeline_winloss.html")
    fig.show()


def plot_language_performance(winloss_df):
    """Plot performance by language."""
    if winloss_df is None:
        print("⚠️  No win/loss data found")
        return
    
    # Melt the dataframe for plotting
    df_melted = winloss_df.melt(
        id_vars=['Language'],
        value_vars=['Direct_nDCG', 'Translate_nDCG'],
        var_name='Pipeline',
        value_name='nDCG@10'
    )
    
    df_melted['Pipeline'] = df_melted['Pipeline'].str.replace('_nDCG', '')
    
    fig = px.bar(
        df_melted,
        x='Language',
        y='nDCG@10',
        color='Pipeline',
        barmode='group',
        title='Performance by Language',
        color_discrete_map={'Direct': '#1f77b4', 'Translate': '#ff7f0e'}
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        height=500
    )
    
    fig.write_html('results/language_performance.html')
    print("✅ Saved: results/language_performance.html")
    fig.show()


def print_summary_stats(summary_df, winloss_df):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    # Best model per pipeline
    for pipeline in summary_df['pipeline'].unique():
        pipeline_data = summary_df[summary_df['pipeline'] == pipeline]
        if 'avg_ndcg@10' in pipeline_data.columns:
            best = pipeline_data.loc[pipeline_data['avg_ndcg@10'].idxmax()]
            print(f"\n{pipeline.upper()} Pipeline:")
            print(f"  Best Model: {best['model']}")
            print(f"  nDCG@10: {best['avg_ndcg@10']:.4f}")
    
    # Overall best
    if 'avg_ndcg@10' in summary_df.columns:
        overall_best = summary_df.loc[summary_df['avg_ndcg@10'].idxmax()]
        print(f"\nOverall Best:")
        print(f"  Model: {overall_best['model']}")
        print(f"  Pipeline: {overall_best['pipeline']}")
        print(f"  nDCG@10: {overall_best['avg_ndcg@10']:.4f}")
    
    # Win/loss summary
    if winloss_df is not None:
        print(f"\nPipeline Win/Loss:")
        win_counts = winloss_df['Winner'].value_counts()
        for pipeline, count in win_counts.items():
            percentage = (count / len(winloss_df)) * 100
            print(f"  {pipeline}: {count}/{len(winloss_df)} ({percentage:.1f}%)")
        
        # Average margin
        avg_margin = winloss_df['Margin'].mean()
        print(f"\nAverage Margin: {avg_margin:.4f}")


def export_latex_table(summary_df):
    """Export results as LaTeX table for thesis."""
    if 'avg_ndcg@10' not in summary_df.columns:
        return
    
    # Select key columns
    cols = ['model', 'pipeline', 'avg_ndcg@10', 'avg_mrr']
    if all(col in summary_df.columns for col in cols):
        table_df = summary_df[cols].copy()
        table_df.columns = ['Model', 'Pipeline', 'nDCG@10', 'MRR']
        
        # Format numbers
        table_df['nDCG@10'] = table_df['nDCG@10'].apply(lambda x: f"{x:.4f}")
        table_df['MRR'] = table_df['MRR'].apply(lambda x: f"{x:.4f}")
        
        # Export to LaTeX
        latex = table_df.to_latex(index=False)
        
        with open('results/results_table.tex', 'w') as f:
            f.write(latex)
        
        print("✅ Saved: results/results_table.tex")


def main():
    parser = argparse.ArgumentParser(description="Compare evaluation results")
    parser.add_argument('--no-plots', action='store_true', help='Skip interactive plots')
    parser.add_argument('--latex', action='store_true', help='Export LaTeX table')
    args = parser.parse_args()
    
    print("Loading results...")
    summary_df, winloss_df = load_results()
    
    if summary_df is None:
        return
    
    print(f"Found {len(summary_df)} result rows")
    
    # Print summary
    print_summary_stats(summary_df, winloss_df)
    
    # Generate plots
    if not args.no_plots:
        print("\nGenerating visualizations...")
        plot_model_comparison(summary_df)
        plot_pipeline_winloss(winloss_df)
        plot_language_performance(winloss_df)
    
    # Export LaTeX
    if args.latex:
        print("\nExporting LaTeX table...")
        export_latex_table(summary_df)
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
