# 修复节点 UI 显示参数与实际裁切不一致的问题

**用户痛点**：
“节点里显示的裁切图的尺寸（UI 面板上的数字）和实际裁切的图的尺寸（输出图）对不上啊？”
-   **UI 面板显示**：`crop_w: 983`, `crop_h: 1536`。
-   **实际输出（中间预览）**：`864 x 1536`。
-   **绿框预览（左侧）**：绿框看起来是 983（比较宽），但用户之前投诉它“宽了”。

**根本原因**：
ComfyUI 的 UI Widget（即面板上的数字输入框）通常是**输入控件**。
在后端 Python 代码中修改了 `crop_w` 和 `crop_h`（例如从 983 修正为 864）之后，**这个修改默认是不会反向同步回前端 UI 的**！
后端计算出了 864，用了 864 裁切，输出了 864 的图。
但前端 UI 上依然停留着运行前的状态（983）。
这就导致了用户看到的“对不上”。

**解决方案：向前端发送更新指令**

我需要在 `RatioCropNode` 中，显式地将修正后的参数（864, 1536 等）返回给前端，强制更新 UI 上的数字。

ComfyUI 支持通过返回 `{"ui": {"widget_values": ...}}` 来更新前端 Widget 的值。

**计划步骤**：
1.  **修改返回值结构**：
    在 `RatioCropNode.crop` 的返回值字典中，增加 `ui` 字段。
2.  **构造 Widget 更新数据**：
    将最终计算出的 `final_x, final_y, final_w, final_h` 放入更新列表。
    注意：需要确认 Widget 的顺序。
    `RatioCropNode` 的 Widget 顺序通常是：`image, ratio, crop_x, crop_y, crop_w, crop_h`。
    或者通过 Widget 名称映射（如果前端支持）。
    通常 ComfyUI 只能按顺序更新，或者不更新。
    但是，如果我在 `RETURN_TYPES` 里已经返回了 `INT` 类型的 `crop_x` 等，这只是数据流输出，不影响 UI。
    **关键点**：必须在返回的字典里包含 `ui` 键。

**Widget 顺序确认**：
```python
            "required": {
                "image": ...,
                "ratio": ...,
                "crop_x": ...,
                "crop_y": ...,
                "crop_w": ...,
                "crop_h": ...,
            },
```
顺序是：
0: image (upload)
1: ratio (combo)
2: crop_x (int)
3: crop_y (int)
4: crop_w (int)
5: crop_h (int)

**更新逻辑**：
```python
        return {
            "ui": {
                "images": [...], # 预览图
                "widget_values": [
                    image,        # 0: 保持原样
                    ratio,        # 1: 保持原样
                    int(final_x), # 2: 更新 x
                    int(final_y), # 3: 更新 y
                    int(final_w), # 4: 更新 w
                    int(final_h)  # 5: 更新 h
                ]
            },
            "result": (...)
        }
```
**风险提示**：
ComfyUI 的某些版本或自定义节点可能对 `widget_values` 的支持有限，或者会触发循环刷新。
但这是解决“UI 数字不更新”的唯一标准方法。

**特别注意**：
如果 batch size > 1，UI 只能显示一个值。我们通常显示第一个图片的参数。

通过这个修改，当节点运行完毕后，UI 面板上的 `983` 会自动跳变为 `864`，用户就会明白“哦，原来系统帮我修正了”，而不会觉得“对不上”。

