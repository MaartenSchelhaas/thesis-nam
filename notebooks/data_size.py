from na2m.data.compas import CompasDataset

ds = CompasDataset()
df = ds.load(r"C:\Users\maart\OneDrive\Documenten\Universiteit\Scriptie\python_repo\thesis-nam\datasets\raw\compas-scores-two-years.csv")
X, y, feature_meta = ds.preprocess(df)

print(f"N = {len(y)}")
print(f"positive rate = {y.mean():.4f}")