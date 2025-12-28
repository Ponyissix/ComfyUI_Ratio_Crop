import os
import torch
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageDraw, ImageFont
import folder_paths


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
    
    # 情况A: 传入了 Tensor (来自外部输入 mask)
    if brush_mask_tensor_in is not None:
        print(f"[RatioCropNode] core_crop_logic processing brush_mask_tensor_in. Shape: {brush_mask_tensor_in.shape}")
        if isinstance(brush_mask_tensor_in, torch.Tensor):
            mask_t = brush_mask_tensor_in.cpu()
            # 如果是 batch > 1，只取第一张
            if mask_t.ndim > 2 and mask_t.shape[0] > 1:
                mask_t = mask_t[0]
            
            mask_np = mask_t.numpy().squeeze()
            # 确保是 2D (H, W)
            if mask_np.ndim > 2:
                mask_np = mask_np[0]

            print(f"[RatioCropNode] Converted mask to numpy shape: {mask_np.shape}")

            mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8), mode="L")
            mask_pil = mask_pil.resize((img_w, img_h), Image.NEAREST) # 确保尺寸一致
            mask_pil = mask_pil.crop((x, y, x + w, y + h))
            mask_np = np.array(mask_pil).astype(np.float32) / 255.0
            final_brush_mask_tensor = torch.from_numpy(mask_np)[None,]
            print(f"[RatioCropNode] Final processed mask tensor shape: {final_brush_mask_tensor.shape}, max value: {final_brush_mask_tensor.max()}")
            
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
                "image_input": ("IMAGE",),
                "brush_mask_path": ("STRING", {"default": "", "multiline": False}),
                "mask": ("MASK",),
                "padding": ("INT", {"default": 20, "min": 0, "max": 100, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "MASK", "CROP_DATA", "INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("original_image", "cropped_image", "crop_mask", "brush_mask", "crop_data", "crop_x", "crop_y", "crop_w", "crop_h", "ratio_str")
    FUNCTION = "crop"
    CATEGORY = "Custom/Crop"
    OUTPUT_NODE = True

    @classmethod
    def VALIDATE_INPUTS(s, **kwargs): return True
    @classmethod
    def validate_input(s, input_types): return True

    def crop(self, image, ratio, crop_x, crop_y, crop_w, crop_h, brush_mask_path="", mask=None, image_input=None, padding=20):
        if mask is not None:
            print(f"[RatioCropNode] Input mask shape: {mask.shape}, dtype: {mask.dtype}, min: {mask.min()}, max: {mask.max()}")
        if image_input is not None:
             print(f"[RatioCropNode] Input image_input shape: {image_input.shape}")

        # 如果有输入图片，优先使用输入图片
        if image_input is not None:
            batch_results = []
            # 遍历 batch
            for i in range(image_input.shape[0]):
                img_tensor = image_input[i]
                img_np = 255. * img_tensor.cpu().numpy()
                image_pil = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8)).convert("RGB")
                
                # 处理 mask
                current_mask = None
                if mask is not None:
                    # 如果 mask 是 batch，尝试对应
                    if mask.shape[0] == image_input.shape[0]:
                        current_mask = mask[i]
                        print(f"[RatioCropNode] Using mask[{i}] for image[{i}]")
                    # 如果 mask 是 batch=1，或者是 2D，使用第一个
                    elif mask.ndim == 2 or (mask.ndim > 0 and mask.shape[0] == 1):
                         if mask.ndim > 2:
                             current_mask = mask[0]
                         else:
                             current_mask = mask
                         print(f"[RatioCropNode] Broadcasting single mask to image[{i}]")
                    # 否则 fallback (可能不匹配)
                    else:
                        if mask.ndim > 2:
                             current_mask = mask[0]
                        else:
                             current_mask = mask
                        print(f"[RatioCropNode] Fallback: Using mask[0] for image[{i}] (Batch mismatch: mask {mask.shape} vs img {image_input.shape})")

                # --- 自动检测 Bounding Box 逻辑 ---
                # 如果传入了 Mask，且 crop_w/crop_h 看起来未设置 (例如为默认值 512，或者用户传入了 0/1)
                # 我们尝试从 Mask 计算 Bounding Box
                # 注意：默认值是 512，如果图片很大，512 可能也是有效的裁切。
                # 但如果用户连接了 image_input 和 mask，通常期望是自动裁切 mask 区域。
                # 为了不破坏手动裁切功能，我们仅当检测到 "可能是未设置" 的情况时才覆盖。
                # 或者，我们可以添加一个逻辑：如果 crop_w <= 1 且 crop_h <= 1 (这通常是不正常的)，则自动计算。
                # 但考虑到默认值是 512，我们很难区分用户是否真的想要 512。
                # 更好的策略：如果 crop_w 和 crop_h 都是 512 (默认值) 且 x=0, y=0，我们假设用户没有手动操作。
                # 或者更激进一点：只要连接了 Mask，且 image_input 存在，我们就优先使用 Mask 的 bbox？
                # 不，这样会覆盖用户的手动调整。
                # 妥协方案：检查 Mask 的 bbox，如果用户没有手动设置（假设 crop_w/h 为默认值或极小值），则使用 Mask bbox。
                
                # 这里的逻辑是：如果用户没有在 UI 上画框 (因为看不到图)，那么 crop 参数可能是默认值。
                # 让我们计算 Mask 的 bbox，并暂时使用它。
                # 但是 core_crop_logic 需要传入 crop_x 等。
                
                # 计算 Mask bbox
                use_mask_bbox = False
                mask_bbox = None
                preview_mask = None # Capture mask for preview

                if current_mask is not None:
                    # current_mask is tensor (H, W) or (1, H, W)
                    m_np = current_mask.cpu().numpy().squeeze()
                    preview_mask = current_mask # Save for preview
                    if m_np.max() > 0:
                        rows = np.any(m_np, axis=1)
                        cols = np.any(m_np, axis=0)
                        y_min, y_max = np.where(rows)[0][[0, -1]]
                        x_min, x_max = np.where(cols)[0][[0, -1]]
                        
                        # 只有当 Mask 有效时
                        mask_bbox = (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)
                        
                        # 判断是否应该使用 Mask bbox
                        # 如果 crop_w 和 crop_h 看起来像默认值或者极小值
                        # 默认值是 512。如果图片尺寸不是 512x512，且 crop 是 0,0,512,512，很可能是默认值。
                        # 或者如果 crop_w <= 1 (如日志中的情况)
                        if crop_w <= 1 and crop_h <= 1:
                            use_mask_bbox = True
                            print("[RatioCropNode] Detected unset crop dimensions (<=1), using Mask Bounding Box.")
                        elif crop_x == 0 and crop_y == 0 and crop_w == 512 and crop_h == 512:
                            # 默认值检测
                             use_mask_bbox = True
                             print("[RatioCropNode] Detected default crop parameters (0,0,512,512), prioritizing Mask Bounding Box.")

                final_x, final_y, final_w, final_h = crop_x, crop_y, crop_w, crop_h
                
                if use_mask_bbox and mask_bbox:
                    bx, by, bw, bh = mask_bbox
                    
                    # Apply Padding (Percentage)
                    # padding 是百分比 0-100
                    base_size = max(bw, bh)
                    pad_px = int(base_size * (padding / 100.0))
                    
                    bx -= pad_px
                    by -= pad_px
                    bw += pad_px * 2
                    bh += pad_px * 2
                    
                    # Clamp to image boundaries
                    img_h_val, img_w_val = img_tensor.shape[1], img_tensor.shape[2] # C, H, W? No, ComfyUI Image is [B, H, W, C]
                    # Wait, img_tensor is [H, W, C] because we iterated batch
                    img_h_val, img_w_val = img_tensor.shape[0], img_tensor.shape[1]
                    
                    bx = max(0, bx)
                    by = max(0, by)
                    if bx + bw > img_w_val: bw = img_w_val - bx
                    if by + bh > img_h_val: bh = img_h_val - by
                    
                    # --- Ratio Adjustment Logic ---
                    target_r = None
                    
                    if ratio == "Free":
                        pass # No adjustment
                    elif ratio == "Smart":
                        # Calculate best standard ratio
                        current_r = bw / bh
                        r_map = {"1:1": 1.0, "3:4": 0.75, "4:3": 1.333, "9:16": 0.5625, "16:9": 1.778, "21:9": 2.333}
                        # Find closest
                        best_r_name = min(r_map.keys(), key=lambda k: abs(r_map[k] - current_r))
                        target_r = r_map[best_r_name]
                        print(f"[RatioCropNode] Smart ratio detected: {best_r_name} ({target_r}) for initial {current_r:.2f}")
                    else:
                        # Fixed ratio
                        r_map = {"1:1": 1.0, "3:4": 0.75, "4:3": 1.333, "9:16": 0.5625, "16:9": 1.778, "21:9": 2.333}
                        if ratio in r_map:
                            target_r = r_map[ratio]

                    if target_r is not None:
                        current_r = bw / bh
                        
                        # Expand to fit ratio
                        if current_r < target_r: 
                            # Current is too tall/narrow -> increase width
                            new_w = bh * target_r
                            diff = new_w - bw
                            bx -= diff / 2
                            bw = new_w
                        else: 
                            # Current is too wide -> increase height
                            new_h = bw / target_r
                            diff = new_h - bh
                            by -= diff / 2
                            bh = new_h
                        
                        # Clamp again after expansion
                        bx = max(0, bx)
                        by = max(0, by)
                        if bx + bw > img_w_val: bw = img_w_val - bx
                        if by + bh > img_h_val: bh = img_h_val - by

                    final_x, final_y, final_w, final_h = bx, by, bw, bh
                    print(f"[RatioCropNode] Auto-calculated bbox from mask with padding {padding}%: x={final_x}, y={final_y}, w={final_w}, h={final_h}")

                # 调用核心逻辑
                res = core_crop_logic(image_pil, final_x, final_y, final_w, final_h, brush_mask_path=brush_mask_path, brush_mask_tensor_in=current_mask, ratio=ratio)
                batch_results.append(res)
            
            # --- Preview Generation ---
            # Use the first image in the batch for preview
            preview_img = batch_results[0][0] # original tensor [1, H, W, C] is returned as first element? No, core_crop_logic returns (original_tensor, ...)
            # Let's verify return of core_crop_logic
            # return (original_tensor, image_tensor, crop_mask_tensor, final_brush_mask_tensor, crop_data, x, y, w, h, ratio_str)
            # original_tensor is [1, H, W, C] or [1, C, H, W]? It's from_numpy(np_array) where np_array is (H,W,C) usually for PIL
            # PIL -> np.array is (H, W, 3)
            # torch.from_numpy -> (H, W, 3)
            # [None,] -> (1, H, W, 3)
            # So standard ComfyUI format.
            
            preview_tensor = batch_results[0][0][0] # (H, W, 3)
            preview_np = 255. * preview_tensor.cpu().numpy()
            preview_pil = Image.fromarray(np.clip(preview_np, 0, 255).astype(np.uint8)).convert("RGB")
            
            # Draw Mask Overlay
            # Note: We use 'preview_mask' which is the full-size mask of the first image in batch
            # NOT 'batch_results[0][3]' which is the cropped mask!
            mask_tensor = preview_mask 
            if mask_tensor is not None:
                # Ensure mask is (H, W)
                m_t = mask_tensor.cpu()
                while m_t.ndim > 2: m_t = m_t.squeeze(0)
                
                if m_t.shape == (preview_pil.size[1], preview_pil.size[0]): # H, W match
                     mask_np = m_t.numpy()
                     
                     # Force mask to be binary 0 or 1 to avoid faint edges
                     mask_np = (mask_np > 0.05).astype(np.float32)

                     # Red mask with alpha
                     mask_overlay = Image.new("RGBA", preview_pil.size, (255, 0, 0, 0))
                     # mask_np is 0 or 1.
                     # 1 -> 230 (approx 90% opacity) for strong solid red look matching manual mode
                     mask_alpha = (mask_np * 230).astype(np.uint8) 
                     mask_layer = Image.fromarray(mask_alpha, mode="L")
                     mask_overlay.putalpha(mask_layer)
                     
                     preview_pil = preview_pil.convert("RGBA")
                     preview_pil.alpha_composite(mask_overlay)
                     preview_pil = preview_pil.convert("RGB")
            
            # --- Dimming Overlay (Darken area OUTSIDE crop box) ---
            bx, by, bw, bh = batch_results[0][5], batch_results[0][6], batch_results[0][7], batch_results[0][8]
            
            # Create a black layer with ~50% opacity
            dim_overlay = Image.new("RGBA", preview_pil.size, (0, 0, 0, 128))
            
            # Create a mask for the dimming layer (transparent inside crop box)
            # 0 = transparent (cut out), 255 = opaque (keep dark)
            dim_mask = Image.new("L", preview_pil.size, 255)
            draw_dim = ImageDraw.Draw(dim_mask)
            draw_dim.rectangle([bx, by, bx+bw, by+bh], fill=0)
            
            # Let's just create the dim layer with the hole and composite it
            dim_overlay.putalpha(dim_mask.point(lambda x: 128 if x > 0 else 0))
            
            preview_pil = preview_pil.convert("RGBA")
            preview_pil.alpha_composite(dim_overlay)
            preview_pil = preview_pil.convert("RGB")

            # Draw Box and Dimensions Text
            draw = ImageDraw.Draw(preview_pil)
            bx, by, bw, bh = batch_results[0][5], batch_results[0][6], batch_results[0][7], batch_results[0][8]
            
            # Calculate dynamic line width based on image size
            # e.g., 3584px -> ~6px. 512px -> 2px.
            line_width = max(2, int(min(preview_pil.size) / 600))
            
            draw.rectangle([bx, by, bx+bw, by+bh], outline="#00FF00", width=line_width)
            
            # Draw Text "WxH"
            text_str = f"{int(bw)}x{int(bh)}"
            
            # --- Dynamic Font Loading ---
            font_size = max(24, int(preview_pil.size[0] / 40)) # e.g. 1000px -> 25px, 4000px -> 100px
            font = None
            try:
                # Try common system fonts
                font_names = ["/System/Library/Fonts/Helvetica.ttc", "arial.ttf", "DejaVuSans.ttf"]
                for fn in font_names:
                    try:
                        font = ImageFont.truetype(fn, size=font_size)
                        break
                    except:
                        continue
            except:
                pass
                
            if font is None:
                # Fallback to default (tiny) if nothing else works, but usually on macOS Helvetica works
                font = ImageFont.load_default()

            # Position above the box
            text_x = bx
            text_y = max(0, by - font_size - 5)
            
            # Draw with shadow
            draw.text((text_x+2, text_y+2), text_str, fill="black", font=font) # shadow
            draw.text((text_x, text_y), text_str, fill="#00FF00", font=font) # text

            # Save Preview
            import random
            filename = f"ratio_crop_preview_{random.randint(0, 1000000)}.png"
            preview_dir = folder_paths.get_temp_directory()
            preview_path = os.path.join(preview_dir, filename)
            preview_pil.save(preview_path)
            
            # 聚合结果
            # core_crop_logic 返回: (original_tensor, image_tensor, crop_mask_tensor, final_brush_mask_tensor, crop_data, x, y, w, h, ratio_str)
            
            # Tensor 列表
            orig_tensors = [r[0] for r in batch_results]
            crop_tensors = [r[1] for r in batch_results]
            crop_mask_tensors = [r[2] for r in batch_results]
            brush_mask_tensors = [r[3] for r in batch_results]
            
            final_orig = torch.cat(orig_tensors, dim=0)
            final_crop = torch.cat(crop_tensors, dim=0)
            final_crop_mask = torch.cat(crop_mask_tensors, dim=0)
            final_brush_mask = torch.cat(brush_mask_tensors, dim=0)
            
            # 对于非 Tensor 数据，取第一个即可 (假设 batch 内参数一致)
            first_res = batch_results[0]
            
            return {"ui": {"images": [{"filename": filename, "subfolder": "", "type": "temp"}]}, "result": (final_orig, final_crop, final_crop_mask, final_brush_mask, first_res[4], first_res[5], first_res[6], first_res[7], first_res[8], first_res[9])}
            
        else:
            # 原有逻辑：从文件加载
            image_path = folder_paths.get_annotated_filepath(image)
            i = Image.open(image_path)
            i = ImageOps.exif_transpose(i)
            if i.mode == 'I': i = i.point(lambda i: i * (1 / 255))
            image_pil = i.convert("RGB")
            
            return core_crop_logic(image_pil, crop_x, crop_y, crop_w, crop_h, brush_mask_path=brush_mask_path, brush_mask_tensor_in=mask, ratio=ratio)

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

# --- 新增节点：批量图片加载器 ---
class RatioBatchLoader:
    OUTPUT_IS_LIST = (True, True) # 允许输出列表，以支持不同尺寸的图片批量处理

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory_path": ("STRING", {"default": "", "multiline": False}),
                "sort_method": (["name", "date"], {"default": "name"}),
                "method": (["Resize to First", "Keep Original"], {"default": "Resize to First"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "masks")
    FUNCTION = "load_batch"
    CATEGORY = "Custom/Crop"

    def load_batch(self, directory_path, sort_method, method):
        if not directory_path or not directory_path.strip():
            # Return empty list
            return ([], [])

        # Handle path: if it's relative to input dir, make it absolute
        if not os.path.isabs(directory_path):
            input_dir = folder_paths.get_input_directory()
            full_path = os.path.join(input_dir, directory_path)
        else:
            full_path = directory_path

        if not os.path.exists(full_path):
            print(f"[RatioBatchLoader] Directory not found: {full_path}")
            return ([], [])

        # Scan for images
        valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff'}
        files = [f for f in os.listdir(full_path) if os.path.splitext(f)[1].lower() in valid_extensions]
        
        if not files:
            print(f"[RatioBatchLoader] No images found in: {full_path}")
            return ([], [])

        # Sort files
        if sort_method == "name":
            files.sort()
        elif sort_method == "date":
            files.sort(key=lambda f: os.path.getmtime(os.path.join(full_path, f)))

        image_list = []
        mask_list = []
        
        # Load first image to determine target size (only for Resize mode)
        target_w, target_h = None, None
        
        if method == "Resize to First":
            first_img_path = os.path.join(full_path, files[0])
            try:
                i = Image.open(first_img_path)
                i = ImageOps.exif_transpose(i)
                if i.mode == 'I': i = i.point(lambda i: i * (1 / 255))
                first_pil = i.convert("RGB")
                target_w, target_h = first_pil.size
                print(f"[RatioBatchLoader] Batch target size: {target_w}x{target_h} (from {files[0]})")
            except Exception as e:
                print(f"[RatioBatchLoader] Failed to load first image: {e}")
                return ([], [])

        for fname in files:
            fpath = os.path.join(full_path, fname)
            try:
                img = Image.open(fpath)
                img = ImageOps.exif_transpose(img)
                if img.mode == 'I': img = img.point(lambda i: i * (1 / 255))
                
                # Resize if necessary (Only in Resize mode)
                if method == "Resize to First" and img.size != (target_w, target_h):
                    print(f"[RatioBatchLoader] Resizing {fname} from {img.size} to {target_w}x{target_h}")
                    img = img.resize((target_w, target_h), Image.LANCZOS)
                
                img_rgb = img.convert("RGB")
                img_np = np.array(img_rgb).astype(np.float32) / 255.0
                
                # Convert to tensor [1, H, W, C]
                img_tensor = torch.from_numpy(img_np)[None,]
                image_list.append(img_tensor)
                
                # Handle Mask (Alpha channel) if exists
                if 'A' in img.getbands():
                    mask = img.getchannel('A')
                    if method == "Resize to First":
                        mask = mask.resize((target_w, target_h), Image.LANCZOS)
                    mask_np = np.array(mask).astype(np.float32) / 255.0
                    mask_list.append(torch.from_numpy(mask_np)[None,])
                else:
                    # Default white mask (fully visible)
                    curr_w, curr_h = img.size
                    mask_list.append(torch.ones((1, curr_h, curr_w), dtype=torch.float32))
                    
            except Exception as e:
                print(f"[RatioBatchLoader] Failed to load {fname}: {e}")
                continue

        if not image_list:
             return ([], [])

        print(f"[RatioBatchLoader] Loaded batch of {len(image_list)} images. Method: {method}")

        if method == "Resize to First":
            # Stack into single batch tensor [B, H, W, C]
            # But since OUTPUT_IS_LIST is True, we must return a list containing this single batch tensor
            batch_images = torch.cat(image_list, dim=0) # [B, H, W, C]
            batch_masks = torch.cat(mask_list, dim=0)   # [B, H, W]
            return ([batch_images], [batch_masks])
        else:
            # Keep Original: Return list of individual tensors
            # Each element is [1, H, W, C]
            return (image_list, mask_list)

NODE_CLASS_MAPPINGS = {
    "RatioCropNode": RatioCropNode,
    "RatioMergeNode": RatioMergeNode,
    "RatioBatchLoader": RatioBatchLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RatioCropNode": "Ratio Crop (Manual)",
    "RatioMergeNode": "Ratio Merge Image",
    "RatioBatchLoader": "Ratio Batch Loader",
}
