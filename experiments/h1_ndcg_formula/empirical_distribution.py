from omnirec.recsys_data_set import RecSysDataSet
from omnirec.data_loaders.datasets import DataSet
from omnirec.preprocess.split import UserHoldout

dataset = RecSysDataSet.use_dataloader(DataSet.MovieLens100K)

# Gleiche Split-Parameter wie im Smoke-Test-Log beobachtet (validation_size=0.1, test_size=0.1)
# Diese Werte ggf. an deine tatsächliche config.toml anpassen!
splitter = UserHoldout(validation_size=0.1, test_size=0.1)
split_dataset = splitter.process(dataset)

test_df = split_dataset._data.test
test_counts_per_user = test_df.groupby("user").size()

print("Gesamtzahl User im Test-Set:", len(test_counts_per_user))
print("\nVerteilung der Anzahl relevanter Test-Items pro User:")
print(test_counts_per_user.describe())

for k in [5, 10, 20]:
    n_below_k = (test_counts_per_user < k).sum()
    pct = 100 * n_below_k / len(test_counts_per_user)
    print(f"\nUser mit weniger als {k} Test-Items: {n_below_k} von {len(test_counts_per_user)} ({pct:.1f}%)")