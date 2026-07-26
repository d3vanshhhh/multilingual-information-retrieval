from datasets import load_dataset

languages = ['en', 'de', 'hi']

print('Query Dataset Statistics')
print('=' * 60)

total = 0
for lang in languages:
    queries = load_dataset('ellamind/wikipedia-2023-11-retrieval-multilingual-queries', lang, split='test')
    count = len(queries)
    total += count
    print(f'{lang.upper()}: {count} queries')

print('=' * 60)
print(f'Total queries across all languages: {total}')
