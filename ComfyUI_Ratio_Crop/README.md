# ComfyUI Ratio Crop Node

这是一个带有交互式界面的 ComfyUI 节点，允许用户在上传的图片上画框并进行裁剪。

## 功能
- **内置图片上传**：无需连接 LoadImage 节点，直接在节点内上传。
- **弹窗编辑器**：点击按钮打开大图编辑器，支持精确画框和**位置调整**。
- **比例约束**：支持 1:1, 3:4, 4:3 和自由比例 (Free) 裁剪。
- **结果预览**：节点界面直接显示**裁剪后的结果**，所见即所得。
- **双输出**：同时输出 `original_image` (原图) 和 `cropped_image` (裁剪图)。

## 安装
1. 将 `ComfyUI_Ratio_Crop` 文件夹复制到你的 ComfyUI 安装目录下的 `custom_nodes` 文件夹中。
   例如：`ComfyUI/custom_nodes/ComfyUI_Ratio_Crop/`
2. 重启 ComfyUI。

## 使用方法
1. 在 ComfyUI 中双击搜索 **"Ratio Crop Image"**。
2. 点击 **image** 按钮上传或选择图片。
3. **编辑裁剪**：
   - 点击 **"选定裁切范围"** 按钮，或者**直接点击节点上的图片**。
   - 在全屏编辑器中，拖拽画框或调整位置。
   - 点击 **"确认裁切"**。
   - 节点上会立即显示裁剪后的预览图。
4. 连接输出：
   - `original_image`: 原始未裁剪的图片。
   - `cropped_image`: 裁剪后的图片。
5. 点击 **Queue Prompt** 执行。

## 参数说明
- **image**: 选择或上传图片。
- **ratio**: 裁剪比例约束（在编辑器中选择会自动同步）。
- **crop_x, crop_y, crop_w, crop_h**: 裁剪区域坐标（会自动根据编辑器操作更新）。
