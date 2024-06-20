from datasets import load_dataset, Audio


fisher = load_dataset('kamilakesbi/fisher_full', num_proc=10)

fisher = fisher.cast_column('audio', Audio(sampling_rate=16000))

fisher.push_to_hub('kamilakesbi/fisher')
