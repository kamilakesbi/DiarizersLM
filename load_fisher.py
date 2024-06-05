from huggingface_hub import snapshot_download

snapshot_download(repo_id="speech-seq2seq/fisher", repo_type="dataset", local_dir="/raid/kamilakesbi/tar_files")
