from datasets import load_dataset
import yaml

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

languages = config['languages']
models = list(config['models'].keys())

print('Dataset Size Analysis')
print('=' * 80)

# Check corpus size
print('\nCORPUS SIZE:')
for lang in languages:
    corpus = load_dataset(config['corpus_dataset'], lang, split='test')
    print(f'  {lang.upper()}: {len(corpus)} documents (using 500 in embeddings)')

# Check queries
print('\nQUERY SIZE:')
total_queries = 0
for lang in languages:
    queries = load_dataset(config['query_dataset'], lang, split='test')
    count = len(queries)
    total_queries += count
    print(f'  {lang.upper()}: {count} queries')

print(f'\n  TOTAL: {total_queries} queries')

# Time estimation
print('\n' + '=' * 80)
print('TIME ESTIMATION')
print('=' * 80)

# Assumptions:
# - Query encoding: ~0.01 seconds per query
# - FAISS search: ~0.001 seconds per query
# - Metric computation: ~0.001 seconds per query
# Total per query: ~0.012 seconds

time_per_query = 0.012  # seconds
pipelines = 2  # direct and translate

# Test mode
test_queries = 5 * len(languages)  # 5 per language
test_time = test_queries * len(models) * pipelines * time_per_query
print(f'\nTEST MODE (5 queries/lang):')
print(f'  Queries: {test_queries}')
print(f'  Models: {len(models)}')
print(f'  Pipelines: {pipelines}')
print(f'  Estimated time: {test_time:.1f} seconds (~{test_time/60:.1f} minutes)')

# Full mode
full_time = total_queries * len(models) * pipelines * time_per_query
print(f'\nFULL MODE (all queries):')
print(f'  Queries: {total_queries}')
print(f'  Models: {len(models)}')
print(f'  Pipelines: {pipelines}')
print(f'  Estimated time: {full_time:.1f} seconds (~{full_time/60:.1f} minutes or ~{full_time/3600:.1f} hours)')

# Recommended: Medium mode
medium_queries_per_lang = 100
medium_queries = medium_queries_per_lang * len(languages)
medium_time = medium_queries * len(models) * pipelines * time_per_query
print(f'\nRECOMMENDED: MEDIUM MODE (100 queries/lang):')
print(f'  Queries: {medium_queries}')
print(f'  Models: {len(models)}')
print(f'  Pipelines: {pipelines}')
print(f'  Estimated time: {medium_time:.1f} seconds (~{medium_time/60:.1f} minutes)')
print(f'  Statistical significance: Much better than 5 queries')
print(f'  Corpus coverage: {medium_queries}/{total_queries} = {medium_queries/total_queries*100:.1f}%')

print('\n' + '=' * 80)
print('RECOMMENDATION:')
print('  Use 100 queries per language for a good balance between:')
print('  - Statistical significance (20x more data than current)')
print('  - Reasonable runtime (~7 minutes)')
print('  - Matches your corpus size (500 docs, so 100 queries is reasonable)')
print('=' * 80)
