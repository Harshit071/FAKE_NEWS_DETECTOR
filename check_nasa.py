import pandas as pd

df = pd.read_csv('news.csv')

# Check for NASA articles
nasa_real = df[(df['text'].str.contains('nasa', case=False, na=False)) & (df['label'] == 'REAL')]
nasa_fake = df[(df['text'].str.contains('nasa', case=False, na=False)) & (df['label'] == 'FAKE')]

print(f"Found {len(nasa_real)} REAL articles containing NASA")
print(f"Found {len(nasa_fake)} FAKE articles containing NASA")

if len(nasa_real) > 0:
    print("\nSample REAL NASA articles:")
    for i, row in nasa_real.head(3).iterrows():
        print(f"Article {i}: {row['text'][:100]}...")
        print(f"Label: {row['label']}")
        print("---")

if len(nasa_fake) > 0:
    print("\nSample FAKE NASA articles:")
    for i, row in nasa_fake.head(3).iterrows():
        print(f"Article {i}: {row['text'][:100]}...")
        print(f"Label: {row['label']}")
        print("---") 