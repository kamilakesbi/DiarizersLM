from datasets import load_dataset


fisher = load_dataset('kamilakesbi/fisher_medium', streaming=True)

print(fisher)