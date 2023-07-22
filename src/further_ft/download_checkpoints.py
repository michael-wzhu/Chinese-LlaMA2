from huggingface_hub import snapshot_download


for repo_id in ["meta-llama/Llama-2-7b-hf"]:
    downloaded = snapshot_download(
        repo_id,
        cache_dir="/your_path/resources",
    )