from collections import Counter
from src.datasets import index_ddd, index_mrl, build_datasets
from src.augmentations import AugPipeline

MRL = r"C:\Users\adith\OneDrive\Documents\engg2112\datasets\MRL"
DDD = r"C:\Users\adith\OneDrive\Documents\engg2112\datasets\DDD"

# First — sanity-check the DDD grouping before you trust any split
ddd = index_ddd(DDD)
groups = Counter(s.subject_id for s in ddd)
print(f"DDD: {len(ddd)} samples, {len(groups)} groups")
print("Top 10 groups:", groups.most_common(10))

# Then the full build
_, _, _, info = build_datasets(mrl_root=MRL, ddd_root=DDD, augment_fn=AugPipeline())
print(info)