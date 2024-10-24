from datasets import load_dataset, Dataset
import pandas as pd

dataset = load_dataset("clips/mfaq", "all_flat")

dataset = dataset['train'].shuffle(seed=42)
print(dataset)

chunks = []
for chunk in dataset.to_pandas(batch_size=1000, batched=True):
    chunks.append(chunk)
    
df = pd.concat(chunks, ignore_index=True)

language_distribution = df['language'].value_counts(normalize=True)

target_size = 2653215
samples_per_language = (language_distribution * target_size).round().astype(int)

downsampled_df = pd.DataFrame()

for language, n_samples in samples_per_language.items():
    print(language)
    print(n_samples)
    language_subset = df[df['language'] == language]
    downsampled_subset = language_subset.sample(n=n_samples, random_state=42)
    downsampled_df = pd.concat([downsampled_df, downsampled_subset])

downsampled_df = downsampled_df.reset_index(drop=True)

# Check if the downsampled dataset has the desired number of rows
print(f"Downsampled dataset size: {downsampled_df.shape[0]}")

grouped = downsampled_df.groupby('language')
for name, group in grouped:
    print(f"Group: {name}")
    group['qa_pairs'] = group.apply(lambda row: [{'question': row['question'], 'answer': row['answer']}], axis=1)
    # group by A and B, and combine the dictionaries in CD
    group_grouped = group.groupby(['domain_id', 'domain'])['qa_pairs'].apply(sum).reset_index()
    group_grouped["num_pairs"] = group_grouped["qa_pairs"].apply(len)
    data = Dataset.from_pandas(group_grouped)
    data.save_to_disk(name+'_shuff')
    data.cache_files
