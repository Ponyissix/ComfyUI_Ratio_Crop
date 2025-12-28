import os
import sys

def check_and_download_model(repo_id, local_dir_name):
    """
    检查并下载模型到插件的 models 目录下。
    使用国内镜像 hf-mirror.com
    """
    # 获取当前文件所在目录 (ComfyUI_Ratio_Crop)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_root = os.path.join(current_dir, "models")
    target_dir = os.path.join(models_root, local_dir_name)
    
    if not os.path.exists(target_dir):
        print(f"[RatioCrop] Model directory not found: {target_dir}")
        os.makedirs(target_dir, exist_ok=True)
    
    # 检查目录下是否有文件 (简单检查)
    if not os.listdir(target_dir):
        print(f"[RatioCrop] Downloading model {repo_id} to {target_dir}...")
        print(f"[RatioCrop] Using mirror: https://hf-mirror.com")
        
        # 设置环境变量使用国内镜像
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=repo_id, 
                local_dir=target_dir, 
                local_dir_use_symlinks=False,
                resume_download=True
            )
            print(f"[RatioCrop] Download completed successfully.")
        except ImportError:
            print("[RatioCrop] Error: 'huggingface_hub' not found. Please install it via 'pip install huggingface_hub'")
            # 尝试回退到 transformers 的 from_pretrained (但这通常不会下载到指定目录结构，而是 cache)
            # 所以这里必须报错提示
            raise ImportError("Please install 'huggingface_hub' to download models automatically.")
        except Exception as e:
            print(f"[RatioCrop] Download failed: {e}")
            raise e
    else:
        print(f"[RatioCrop] Model {repo_id} found at {target_dir}")
        
    return target_dir
