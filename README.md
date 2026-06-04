# RLADLS / FTADLS

Run experiments from YAML:

```bash
python main.py --config config.yaml
```

`experiment.case` controls the experiment:

- `in_domain`
- `cross_dataset_adaptation`
- `both`

Sequence construction is controlled by:

```yaml
sequence:
  construction: group_by_column

dataset_group_cols:
  HDFS: Node_block_id
  BGL: Node_block_id
  TH_1G: Node_block_id
  SP_150MB: Node_block_id
```

With this setting, the code does **not** create new sliding windows. It groups logs by `Node_block_id`, so each preprocessed window becomes one RL episode.
