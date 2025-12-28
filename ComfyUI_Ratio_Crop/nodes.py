import os
import torch
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageDraw
import folder_paths

# 尝试导入 transformers 和 model_utils
try:
    from transformers import (
        GroundingDinoProcessor, GroundingDinoForObjectDetection,
        SamProcessor, SamModel
    )
    from .model_utils import check_and_download_model
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("[RatioCrop] Transformers or model_utils not available. AutoCropSAM node will be disabled.")

# --- 辅助函数：核心裁切逻辑 ---
def core_crop_logic(image, crop_x, crop_y, crop_w, crop_h, brush_mask_path="", brush_mask_tensor_in=None, ratio="Free"):
    """
    核心裁切逻辑，供 RatioCropNode 和 RatioAutoCropSAM 复用。
    image: PIL Image (RGB)
    brush_mask_tensor_in: 如果有直接传入的 tensor (来自 AutoCrop)，则优先使用，否则尝试读取 path
    """
    img_w, img_h = image.size
    
    # 边界检查与修正
    x = max(0, int(crop_x))
    y = max(0, int(crop_y))
    w = int(crop_w)
    h = int(crop_h)
    
    if x + w > img_w: w = img_w - x
    if y + h > img_h: h = img_h - y
    w = max(1, w)
    h = max(1, h)
    
    # 执行裁切
    cropped_img = image.crop((x, y, x + w, y + h))
    
    # 转换 Image Tensor
    image_np = np.array(cropped_img).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np)[None,]
    
    # 转换 Original Tensor
    original_np = np.array(image).astype(np.float32) / 255.0
    original_tensor = torch.from_numpy(original_np)[None,]

    # 1. 创建 Crop Mask (局部全白)
    crop_mask_img = Image.new("L", (w, h), 255)
    crop_mask_np = np.array(crop_mask_img).astype(np.float32) / 255.0
    crop_mask_tensor = torch.from_numpy(crop_mask_np)[None,]

    # 2. 处理 Brush Mask
    final_brush_mask_tensor = None
    
    # 情况A: 传入了 Tensor (来自自动分割)
    if brush_mask_tensor_in is not None:
        # 假设传入的是 (1, H, W) 或 (H, W)，需裁切
        # 这里的 mask 应该是全图尺寸的
        if isinstance(brush_mask_tensor_in, torch.Tensor):
            mask_np = brush_mask_tensor_in.cpu().numpy().squeeze()
            mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8), mode="L")
            mask_pil = mask_pil.resize((img_w, img_h), Image.NEAREST) # 确保尺寸一致
            mask_pil = mask_pil.crop((x, y, x + w, y + h))
            mask_np = np.array(mask_pil).astype(np.float32) / 255.0
            final_brush_mask_tensor = torch.from_numpy(mask_np)[None,]
            
    # 情况B: 传入了路径 (来自手动涂抹)
    elif brush_mask_path and brush_mask_path.strip():
        try:
            input_dir = folder_paths.get_input_directory()
            mask_file_path = os.path.join(input_dir, brush_mask_path)
            if os.path.exists(mask_file_path):
                brush_img = Image.open(mask_file_path)
                if brush_img.size != (img_w, img_h):
                    brush_img = brush_img.resize((img_w, img_h), Image.LANCZOS)
                
                if brush_img.mode == 'RGBA':
                    brush_mask_pil = brush_img.split()[3]
                else:
                    brush_mask_pil = brush_img.convert("L")
                
                brush_mask_pil = brush_mask_pil.crop((x, y, x + w, y + h))
                brush_mask_np = np.array(brush_mask_pil).astype(np.float32) / 255.0
                final_brush_mask_tensor = torch.from_numpy(brush_mask_np)[None,]
        except Exception as e:
            print(f"[RatioCrop] Failed to load brush mask: {e}")

    if final_brush_mask_tensor is None:
        final_brush_mask_tensor = torch.zeros((1, h, w), dtype=torch.float32)

    # 3. 修正 Ratio String
    ratio_str = ratio
    if ratio == "Smart":
        current_ratio = w / h
        ratios = {"1:1": 1.0, "3:4": 3/4, "4:3": 4/3, "9:16": 9/16, "16:9": 16/9, "21:9": 21/9}
        best_r = min(ratios.keys(), key=lambda k: abs(ratios[k] - current_ratio))
        ratio_str = best_r

    crop_data = {
        "crop_x": x, "crop_y": y, "crop_w": w, "crop_h": h,
        "img_width": img_w, "img_height": img_h
    }

    return (original_tensor, image_tensor, crop_mask_tensor, final_brush_mask_tensor, crop_data, x, y, w, h, ratio_str)

# --- 现有节点：手动裁切 ---
class RatioCropNode:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True}),
                "ratio": (["Smart", "1:1", "3:4", "4:3", "9:16", "16:9", "21:9"], {"default": "Smart"}),
                "crop_x": ("INT", {"default": 0, "min": 0, "max": 1000000}),
                "crop_y": ("INT", {"default": 0, "min": 0, "max": 1000000}),
                "crop_w": ("INT", {"default": 512, "min": 1, "max": 1000000}),
                "crop_h": ("INT", {"default": 512, "min": 1, "max": 1000000}),
            },
            "optional": {
                "brush_mask_path": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "MASK", "CROP_DATA", "INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("original_image", "cropped_image", "crop_mask", "brush_mask", "crop_data", "crop_x", "crop_y", "crop_w", "crop_h", "ratio_str")
    FUNCTION = "crop"
    CATEGORY = "Custom/Crop"

    @classmethod
    def VALIDATE_INPUTS(s, **kwargs): return True
    @classmethod
    def validate_input(s, input_types): return True

    def crop(self, image, ratio, crop_x, crop_y, crop_w, crop_h, brush_mask_path=""):
        image_path = folder_paths.get_annotated_filepath(image)
        i = Image.open(image_path)
        i = ImageOps.exif_transpose(i)
        if i.mode == 'I': i = i.point(lambda i: i * (1 / 255))
        image_pil = i.convert("RGB")
        
        return core_crop_logic(image_pil, crop_x, crop_y, crop_w, crop_h, brush_mask_path=brush_mask_path, ratio=ratio)

# --- 新增节点：批量加载器 ---
class RatioBatchLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "", "multiline": False}),
                "limit": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "start_index": ("INT", {"default": 0, "min": 0, "max": 10000}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "LIST")
    RETURN_NAMES = ("images", "filenames")
    FUNCTION = "load_batch"
    CATEGORY = "Custom/Crop"

    def load_batch(self, folder_path, limit, start_index):
        if not os.path.exists(folder_path):
            print(f"[RatioBatchLoader] Folder not found: {folder_path}")
            return (torch.zeros((1, 512, 512, 3)), [])

        # 获取图片文件
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
        files = sorted([f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in valid_exts])
        
        if start_index >= len(files):
            print(f"[RatioBatchLoader] Start index {start_index} out of range (total {len(files)})")
            return (torch.zeros((1, 512, 512, 3)), [])

        selected_files = files[start_index:]
        if limit > 0:
            selected_files = selected_files[:limit]

        if not selected_files:
            return (torch.zeros((1, 512, 512, 3)), [])

        images = []
        filenames = []
        
        # 读取第一张图确定基准尺寸
        first_img_path = os.path.join(folder_path, selected_files[0])
        try:
            first_img = Image.open(first_img_path)
            first_img = ImageOps.exif_transpose(first_img).convert("RGB")
            base_w, base_h = first_img.size
        except Exception as e:
            print(f"[RatioBatchLoader] Error loading first image: {e}")
            return (torch.zeros((1, 512, 512, 3)), [])

        for fname in selected_files:
            fpath = os.path.join(folder_path, fname)
            try:
                img = Image.open(fpath)
                img = ImageOps.exif_transpose(img).convert("RGB")
                
                # 强制缩放到基准尺寸 (ComfyUI Batch 要求尺寸一致)
                if img.size != (base_w, base_h):
                    img = img.resize((base_w, base_h), Image.LANCZOS)
                
                img_np = np.array(img).astype(np.float32) / 255.0
                images.append(img_np)
                filenames.append(fname)
            except Exception as e:
                print(f"[RatioBatchLoader] Failed to load {fname}: {e}")

        if not images:
            return (torch.zeros((1, 512, 512, 3)), [])

        image_batch = torch.from_numpy(np.array(images))
        print(f"[RatioBatchLoader] Loaded {len(images)} images from {folder_path}")
        return (image_batch, filenames)

# --- 新增节点：智能裁切 (OwlViT + SAM) ---
class RatioAutoCropSAM:
    @classmethod
    def INPUT_TYPES(s):
        # 扫描 models/sams 目录
        sams_dir = os.path.join(folder_paths.models_dir, "sams")
        sam_files = []
        if os.path.exists(sams_dir):
            sam_files = [f for f in os.listdir(sams_dir) if f.endswith(".pth") or f.endswith(".pt")]
        
        # 如果没有找到 SAM 模型，提供一个提示
        if not sam_files:
            sam_files = ["No SAM models found in models/sams"]

        # 扫描 models/grounding-dino 目录
        dino_dir = os.path.join(folder_paths.models_dir, "grounding-dino")
        dino_files = []
        if os.path.exists(dino_dir):
            dino_files = [f for f in os.listdir(dino_dir) if f.endswith(".pth") or f.endswith(".pt")]
        
        # 如果没有找到 GroundingDINO 模型，提供一个提示
        if not dino_files:
            dino_files = ["No GroundingDINO models found in models/grounding-dino"]

        return {
            "required": {
                "image": ("IMAGE",),
                "sam_model": (sorted(sam_files),),
                "grounding_dino_model": (sorted(dino_files),),
                "threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "detail_method": (["VITMatte", "PyMatting", "GuidedFilter"], {"default": "VITMatte"}),
                "detail_erode": ("INT", {"default": 6, "min": 0, "max": 100}),
                "detail_dilate": ("INT", {"default": 6, "min": 0, "max": 100}),
                "black_point": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
                "white_point": ("FLOAT", {"default": 0.99, "min": 0.0, "max": 1.0, "step": 0.01}),
                "process_detail": ("BOOLEAN", {"default": True}),
                "prompt": ("STRING", {"default": "subject", "multiline": False}),
                "device": (["cuda", "cpu", "mps"], {"default": "mps"}),
                "max_megapixels": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "cache_model": ("BOOLEAN", {"default": False}),
                "padding": ("INT", {"default": 0, "min": 0, "max": 500}),
                "ratio": (["Smart", "1:1", "3:4", "4:3", "9:16", "16:9", "21:9"], {"default": "Smart"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "MASK", "CROP_DATA", "INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("original_image", "cropped_image", "crop_mask", "sam_mask", "crop_data", "crop_x", "crop_y", "crop_w", "crop_h", "ratio_str")
    FUNCTION = "auto_crop"
    CATEGORY = "Custom/Crop"

    def load_groundingdino(self, model_name, device):
        try:
            import groundingdino.datasets.transforms as T
            from groundingdino.models import build_model
            from groundingdino.util.slconfig import SLConfig
            from groundingdino.util.utils import clean_state_dict
        except ImportError:
            raise ImportError("Please install 'groundingdino-py' to load local .pth models.")

        dino_dir = os.path.join(folder_paths.models_dir, "grounding-dino")
        ckpt_path = os.path.join(dino_dir, model_name)
        
        # ... (配置查找逻辑不变) ...
        if "swinb" in model_name.lower():
            config_name = "GroundingDINO_SwinB.cfg.py"
        else:
            config_name = "GroundingDINO_SwinT_OGC.cfg.py"
            
        config_path = os.path.join(dino_dir, config_name)
        if not os.path.exists(config_path):
            configs = [f for f in os.listdir(dino_dir) if f.endswith(".cfg.py")]
            if configs:
                config_path = os.path.join(dino_dir, configs[0])
            else:
                raise FileNotFoundError(f"Config file not found for {model_name} in {dino_dir}")

        args = SLConfig.fromfile(config_path)
        
        # 修正：GroundingDINO 的 SLConfig 可能会根据 device 参数自动设置 text_encoder_type 等
        # 如果 args.device 被设置为 cuda，但当前环境是 mps，可能会有问题
        # 强制修正 args.device 为 'cpu' 或 'cuda' (GroundingDINO 可能不支持 mps 字符串)
        # 但我们可以在 build_model 后再 to(device)
        # 这里 args.device 主要是为了构建模型时的默认位置
        
        # 强制所有环境先用 CPU 初始化，避免 build_model 内部的 CUDA 检查
        # 除非确实是 CUDA 环境，GroundingDINO 可能有些优化需要初始化时就在 CUDA 上
        if device != "cuda":
            args.device = "cpu"
        else:
            args.device = "cuda"
            
        model = build_model(args)
        
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        model.to(device)
        model.eval()
        return model

    def load_sam(self, model_name, device):
        try:
            from segment_anything import sam_model_registry
        except ImportError:
            raise ImportError("Please install 'segment-anything' to load local .pth models.")

        sam_dir = os.path.join(folder_paths.models_dir, "sams")
        ckpt_path = os.path.join(sam_dir, model_name)
        
        # 推断模型类型
        model_type = "vit_h" # Default
        if "vit_b" in model_name:
            model_type = "vit_b"
        elif "vit_l" in model_name:
            model_type = "vit_l"
        elif "vit_h" in model_name:
            model_type = "vit_h"
            
        # 修复：在 Mac 上加载 CUDA 训练的权重时，需要指定 map_location
        # segment_anything 的 build_sam 内部使用 torch.load，它可能没有正确传递 map_location
        # 为了解决这个问题，我们需要手动 load state dict，然后传给 build_sam
        # 或者更简单：在加载前检查是否是 CUDA 环境
        
        # 如果当前环境不是 CUDA，强制将权重加载到 CPU
        # segment_anything 库允许传入 checkpoint 路径，但它的 build_sam 并没有暴露 map_location 参数
        # 我们必须手动加载 state_dict
        
        if device != "cuda":
            state_dict = torch.load(ckpt_path, map_location="cpu")
            sam = sam_model_registry[model_type](checkpoint=None)
            sam.load_state_dict(state_dict)
        else:
            # 如果是 CUDA 环境，直接让库自己处理
            sam = sam_model_registry[model_type](checkpoint=ckpt_path)
            
        sam.to(device)
        return sam

    def auto_crop(self, image, sam_model, grounding_dino_model, threshold, detail_method, detail_erode, detail_dilate, 
                  black_point, white_point, process_detail, prompt, device, max_megapixels, cache_model, padding, ratio):
        
        # 1. 加载模型 (使用本地库)
        print(f"[RatioAutoCropSAM] Loading local models: {sam_model}, {grounding_dino_model}")
        print(f"[RatioAutoCropSAM] Using device: {device}")
        
        # 加载 GroundingDINO
        dino_model = self.load_groundingdino(grounding_dino_model, device)
        
        # 加载 SAM
        sam_model_instance = self.load_sam(sam_model, device)
        
        # 准备预测器
        from segment_anything import SamPredictor
        sam_predictor = SamPredictor(sam_model_instance)

        # 2. 处理图片
        batch_results = {
            "orig": [], "crop": [], "c_mask": [], "b_mask": [], "data": [], 
            "x": [], "y": [], "w": [], "h": [], "r": []
        }
        
        for i in range(image.shape[0]):
            img_tensor = image[i]
            img_np = 255. * img_tensor.cpu().numpy()
            pil_image = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8)).convert("RGB")
            w, h = pil_image.size

            # --- Step A: GroundingDINO Detection ---
            # 使用 groundingdino 库的 inference 逻辑
            import groundingdino.datasets.transforms as T
            
            # 预处理图片
            transform = T.Compose([
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            image_tensor, _ = transform(pil_image, None)
            image_tensor = image_tensor.to(device)
            
            # 处理 Prompt (GroundingDINO 需要 . 分隔)
            clean_prompts = [p.strip() for p in prompt.split(",") if p.strip()]
            if not clean_prompts: clean_prompts = ["object"]
            dino_prompt = " . ".join(clean_prompts)
            if not dino_prompt.endswith("."): dino_prompt += "."
            
            with torch.no_grad():
                outputs = dino_model(image_tensor[None], captions=[dino_prompt])
            
            logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (NQ, 256)
            boxes = outputs["pred_boxes"].cpu()[0]  # (NQ, 4)
            
            # 过滤
            filt_mask = logits.max(dim=1)[0] > threshold
            boxes_filt = boxes[filt_mask]
            
            # 转换坐标 (cx, cy, w, h) -> (x, y, x, y)
            # boxes_filt 是归一化的 (cx, cy, w, h)
            H, W = pil_image.size[1], pil_image.size[0]
            
            final_boxes = []
            for box in boxes_filt:
                boxes_filt_pixel = box * torch.Tensor([W, H, W, H])
                cx, cy, bw, bh = boxes_filt_pixel
                x1 = cx - bw / 2
                y1 = cy - bh / 2
                x2 = cx + bw / 2
                y2 = cy + bh / 2
                final_boxes.append([x1.item(), y1.item(), x2.item(), y2.item()])
            
            final_boxes = torch.tensor(final_boxes)

            # 调试输出
            if len(final_boxes) > 0:
                print(f"[RatioAutoCropSAM] Detected {len(final_boxes)} objects with threshold {threshold}")
            else:
                print(f"[RatioAutoCropSAM] No objects found above threshold {threshold}.")

            # 如果没检测到，默认全图
            if len(final_boxes) == 0:
                final_box = [0, 0, w, h]
                mask_tensor = torch.zeros((1, h, w))
            else:
                # 合并检测框 (计算所有框的并集)
                min_x = final_boxes[:, 0].min().item()
                min_y = final_boxes[:, 1].min().item()
                max_x = final_boxes[:, 2].max().item()
                max_y = final_boxes[:, 3].max().item()
                
                # --- Step B: SAM Segmentation ---
                sam_predictor.set_image(np.array(pil_image))
                
                # 将所有检测框喂给 SAM
                transformed_boxes = sam_predictor.transform.apply_boxes_torch(final_boxes.to(device), pil_image.size[::-1])
                
                masks, _, _ = sam_predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False,
                )
                
                # masks: (N_boxes, 1, H, W)
                combined_mask = torch.any(masks.squeeze(1), dim=0) # (H, W)
                mask_tensor = combined_mask.float().cpu().unsqueeze(0) # (1, H, W)
                
                # --- Detail Processing (Matting) ---
                if process_detail:
                     m_pil = Image.fromarray((combined_mask.cpu().numpy() * 255).astype(np.uint8))
                     if detail_erode > 0:
                         m_pil = m_pil.filter(ImageFilter.MinFilter(detail_erode))
                     if detail_dilate > 0:
                         m_pil = m_pil.filter(ImageFilter.MaxFilter(detail_dilate))
                     
                     m_np = np.array(m_pil).astype(np.float32) / 255.0
                     combined_mask = torch.from_numpy(m_np) > 0.5
                     mask_tensor = combined_mask.float().unsqueeze(0)

                y_indices, x_indices = torch.where(combined_mask.cpu())
                if len(y_indices) > 0:
                    mask_min_x, mask_max_x = x_indices.min().item(), x_indices.max().item()
                    mask_min_y, mask_max_y = y_indices.min().item(), y_indices.max().item()
                    final_box = [mask_min_x, mask_min_y, mask_max_x - mask_min_x + 1, mask_max_y - mask_min_y + 1]
                else:
                     final_box = [min_x, min_y, max_x - min_x, max_y - min_y]

            # --- Step C: Apply Padding & Ratio ---
            bx, by, bw, bh = final_box
            
            base_size = max(bw, bh)
            pad_px = int(base_size * (padding / 100.0))
            
            bx -= pad_px
            by -= pad_px
            bw += pad_px * 2
            bh += pad_px * 2
            
            bx = max(0, bx)
            by = max(0, by)
            if bx + bw > w: bw = w - bx
            if by + bh > h: bh = h - by
            
            target_r = bw / bh
            if ratio != "Smart" and ratio != "Free":
                r_map = {"1:1": 1.0, "3:4": 0.75, "4:3": 1.333, "9:16": 0.5625, "16:9": 1.778, "21:9": 2.333}
                if ratio in r_map:
                    target_r = r_map[ratio]
                    current_r = bw / bh
                    
                    if current_r < target_r: 
                        new_w = bh * target_r
                        diff = new_w - bw
                        bx -= diff / 2
                        bw = new_w
                    else: 
                        new_h = bw / target_r
                        diff = new_h - bh
                        by -= diff / 2
                        bh = new_h

            res = core_crop_logic(
                pil_image, bx, by, bw, bh, 
                brush_mask_tensor_in=mask_tensor,
                ratio=ratio
            )
            
            batch_results["orig"].append(res[0])
            batch_results["crop"].append(res[1])
            batch_results["c_mask"].append(res[2])
            batch_results["b_mask"].append(res[3])
            last_res = res
            
        final_orig = torch.cat(batch_results["orig"], dim=0)
        final_crop = torch.cat(batch_results["crop"], dim=0)
        final_c_mask = torch.cat(batch_results["c_mask"], dim=0)
        final_b_mask = torch.cat(batch_results["b_mask"], dim=0)
        
        target_h, target_w = final_crop[0].shape[1], final_crop[0].shape[2]
        needs_resize = False
        for i in range(1, final_crop.shape[0]):
            if final_crop[i].shape[1:] != (target_h, target_w):
                needs_resize = True
                break
        
        if needs_resize:
            resized_crops = []
            resized_c_masks = []
            resized_b_masks = []
            
            for i in range(len(batch_results["crop"])):
                c_img = batch_results["crop"][i]
                c_img_p = c_img.permute(0, 3, 1, 2)
                c_img_r = torch.nn.functional.interpolate(c_img_p, size=(target_h, target_w), mode='bilinear', align_corners=False)
                resized_crops.append(c_img_r.permute(0, 2, 3, 1))
                
                c_mask = batch_results["c_mask"][i].unsqueeze(1)
                c_mask_r = torch.nn.functional.interpolate(c_mask, size=(target_h, target_w), mode='nearest')
                resized_c_masks.append(c_mask_r.squeeze(1))
                
                b_mask = batch_results["b_mask"][i].unsqueeze(1)
                b_mask_r = torch.nn.functional.interpolate(b_mask, size=(target_h, target_w), mode='nearest')
                resized_b_masks.append(b_mask_r.squeeze(1))
                
            final_crop = torch.cat(resized_crops, dim=0)
            final_c_mask = torch.cat(resized_c_masks, dim=0)
            final_b_mask = torch.cat(resized_b_masks, dim=0)

        del dino_model, sam_model_instance
        torch.cuda.empty_cache()

        return (final_orig, final_crop, final_c_mask, final_b_mask, last_res[4], last_res[5], last_res[6], last_res[7], last_res[8], last_res[9])


class RatioMergeNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "cropped_image": ("IMAGE",),
                "crop_data": ("CROP_DATA",),
                "feather": ("INT", {"default": 0, "min": 0, "max": 500, "step": 1}),
                "padding": ("INT", {"default": 0, "min": 0, "max": 500, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("merged_image",)
    FUNCTION = "merge"
    CATEGORY = "Custom/Crop"

    def merge(self, original_image, cropped_image, crop_data, feather, padding):
        # 解包参数
        crop_x = crop_data["crop_x"]
        crop_y = crop_data["crop_y"]
        crop_w = crop_data["crop_w"]
        crop_h = crop_data["crop_h"]

        # 1. 处理原始图片 (original_image)
        orig_tensor = original_image[0]
        orig_np = 255. * orig_tensor.cpu().numpy()
        orig_pil = Image.fromarray(np.clip(orig_np, 0, 255).astype(np.uint8)).convert("RGBA")
        
        # 2. 处理裁剪图片 (cropped_image)
        crop_tensor = cropped_image[0]
        crop_np = 255. * crop_tensor.cpu().numpy()
        crop_pil = Image.fromarray(np.clip(crop_np, 0, 255).astype(np.uint8)).convert("RGBA")
        
        # 3. 缩放裁剪图片以匹配目标尺寸
        target_w = max(1, int(crop_w))
        target_h = max(1, int(crop_h))
        
        if crop_pil.size != (target_w, target_h):
            crop_pil = crop_pil.resize((target_w, target_h), Image.LANCZOS)
            
        # 4. 边缘羽化
        if feather > 0 or padding > 0:
            mask = Image.new("L", (target_w, target_h), 0)
            draw = ImageDraw.Draw(mask)
            
            p = min(padding, target_w // 2 - 1, target_h // 2 - 1)
            p = max(0, p)
            
            draw.rectangle((p, p, target_w - p, target_h - p), fill=255)
            
            if feather > 0:
                mask = mask.filter(ImageFilter.GaussianBlur(feather))
            
            r, g, b, a = crop_pil.split()
            a_np = np.array(a).astype(np.float32)
            m_np = np.array(mask).astype(np.float32)
            new_a_np = a_np * (m_np / 255.0)
            new_a = Image.fromarray(new_a_np.astype(np.uint8))
            
            crop_pil.putalpha(new_a)
            
            merged_pil = orig_pil.copy()
            merged_pil.paste(crop_pil, (int(crop_x), int(crop_y)), crop_pil)
            
        else:
            merged_pil = orig_pil.copy()
            merged_pil.paste(crop_pil, (int(crop_x), int(crop_y)))
        
        # 5. 转换回 Tensor (RGB)
        merged_pil = merged_pil.convert("RGB")
        merged_np = np.array(merged_pil).astype(np.float32) / 255.0
        merged_tensor = torch.from_numpy(merged_np)[None,]
        
        return (merged_tensor,)
