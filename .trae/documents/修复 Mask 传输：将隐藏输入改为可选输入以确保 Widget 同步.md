# 终极排查：为什么 Mask 还是没有输出？

用户反馈“还是没有看到正确的蒙版”。结合之前的修改，我们已经确保了：
1.  前端上传成功，拿到了真实文件名。
2.  后端有了读取逻辑。

**唯一的断点**在于：**前端拿到文件名后，是否真的成功告诉了后端？**

## 深度怀疑点
ComfyUI 的 `hidden` widget 有一个特性：**它们可能不会被序列化传输给后端**，或者在前端修改了 hidden widget 的 value 后，没有触发图表的“脏”状态（Dirty State），导致 ComfyUI 认为参数没变，从而不重新发送给后端，甚至不重新执行节点。

**更糟糕的是**：在 `nodes.py` 中定义的 `hidden` 输入，在前端对应的 Widget 可能根本不存在于 `this.widgets` 数组中，或者是一个特殊的属性。
我在 `js/crop_ui.js` 里写的是 `this.widgets.find(w => w.name === "brush_mask_path")`。如果这个 widget 实际上不存在（因为它是 hidden 的），那么 `this.w_mask_path` 就是 `undefined`，赋值操作自然就失效了。

## 验证与修正方案

### 1. 确认 Widget 是否存在
*   在 ComfyUI 中，`hidden` 输入通常不会自动创建前端 Widget。我们需要在前端**手动创建一个隐藏的 Widget** 或者利用 `widgets_values` 数组来传递。
*   **修正策略**：在 `onNodeCreated` 里，检查是否存在 `brush_mask_path` widget。如果不存在，手动添加一个隐藏的 widget，或者直接操作节点的 `widgets_values`（如果节点序列化是基于这个的）。
*   **最佳实践**：将 `brush_mask_path` 定义为普通的 `STRING` widget，但将其 `hidden` 属性设为 `true`，或者将其类型设为 `converted-widget` 等。但在 ComfyUI 扩展开发中，最稳妥的方式是：**定义为普通 Widget，但在前端将其隐藏（computed height = 0 或 hidden flag）**。

### 2. 后端定义调整
目前后端是：
```python
"hidden": { "brush_mask_path": ("STRING", {"default": ""}) }
```
这会导致 ComfyUI 前端**不会**生成这个 widget。
**必须改为**：
```python
"required": { ..., "brush_mask_path": ("STRING", {"default": "", "multiline": False}), ... }
```
然后在前端代码里，找到这个 widget 并把它隐藏起来（`w.type = "hidden"` 或 `w.computeSize = () => [0, -4]`）。这样既保证了数据通道的存在，又不会在 UI 上碍眼。

### 3. 强制图表更新
在赋值后，必须调用：
```javascript
app.graph.setDirtyCanvas(true, true);
```
并且，如果用户没有改变其他参数（只画了 Mask），ComfyUI 可能认为节点不需要重跑。我们需要确保上传 Mask 后，节点的某个参数变了，从而触发执行。修改 `brush_mask_path` 本身就是改变参数，所以只要 Widget 存在且值变了，就没问题。

## 实施计划

1.  **修改 `nodes.py`**：将 `brush_mask_path` 从 `hidden` 移到 `optional` (或 required 但有默认值)，确保前端生成 Widget。
2.  **修改 `js/crop_ui.js`**：
    *   在 `onNodeCreated` 中找到这个 Widget。
    *   将其隐藏（设置 `hidden = true` 或其它 UI 隐藏手段）。
    *   确保上传后正确赋值给这个 Widget。

请确认执行。