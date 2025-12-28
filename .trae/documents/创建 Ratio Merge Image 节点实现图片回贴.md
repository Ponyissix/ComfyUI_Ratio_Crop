# 创建 "Ratio Merge Image" 节点

这个新节点的作用是**逆向操作**：它接收一张原始底图、一张修改后的局部图（比如经过重绘的），以及裁剪时的坐标信息，然后将局部图**无缝贴回**到底图的准确位置上。

## 节点设计

### 输入 (INPUT_TYPES)
1.  **original_image** (`IMAGE`): 原始底图（通常来自 `Ratio Crop` 的 `original_image` 输出）。
2.  **cropped_image** (`IMAGE`): 经过处理后的局部图（比如经过 SD 图生图处理过的）。
3.  **crop_x** (`INT`): 贴回的 X 坐标。
4.  **crop_y** (`INT`): 贴回的 Y 坐标。
5.  **crop_w** (`INT`): 贴回的宽度。
6.  **crop_h** (`INT`): 贴回的高度。
    *   *注：这些坐标参数可以直接从 `Ratio Crop` 节点转换或手动输入，但在 ComfyUI 中，更优雅的方式是让 `Ratio Crop` 输出一个 `CROP_DATA` 绑定包，或者用户手动连接 4 个 INT。为了通用性，我们先做 4 个 INT 输入，并支持转换为 Input 连线。*

### 输出 (RETURN_TYPES)
1.  **merged_image** (`IMAGE`): 合成后的完整大图。

### 核心逻辑
1.  将 `cropped_image` 缩放到 `crop_w x crop_h`（防止处理过程中尺寸发生微变）。
2.  创建一个与 `original_image` 一样大的画布。
3.  将 `original_image` 铺底。
4.  将 `cropped_image` 粘贴到 `(crop_x, crop_y)` 位置。
5.  可选：支持简单的边缘羽化（本次先做直接粘贴，确保位置精确）。

## 实施步骤
1.  修改 `nodes.py`，添加 `RatioMergeNode` 类。
2.  注册新节点。
3.  不需要额外的前端 JS，这是一个纯后端逻辑节点。

请确认执行。