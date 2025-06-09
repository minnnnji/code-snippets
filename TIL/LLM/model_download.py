from huggingface_hub import snapshot_download
snapshot_download(repo_id="naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B",
                  local_dir="./clovax_raw",
                  revision="main")

# snapshot_download(repo_id="Mungert/HyperCLOVAX-SEED-Text-Instruct-0.5B-GGUF",
#                   local_dir="./gguf",
#                   allow_patterns="*.gguf")

