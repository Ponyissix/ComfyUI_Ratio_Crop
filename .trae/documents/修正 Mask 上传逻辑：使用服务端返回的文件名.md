# 重构 Mask 输出逻辑：参考 ComfyUI 原生实现

用户反馈 `brush_mask` 依然输出全黑（无效），并建议参考 ComfyUI 自带的 MaskEditor。

## 问题根源分析
1.  **上传时机与 Widget 更新时机不匹配**：
    *   我们在 `confirmBtn.onclick` 里执行了上传，然后更新 `this.w_mask_path.value`。
    *   但是，ComfyUI 的节点执行（Execution）并不一定会在这个时刻触发，或者说，当用户点击“Queue Prompt”时，前端的 Widget 值是否真的同步到了后端？
    *   **关键点**：`this.w_mask_path` 是一个隐藏的 STRING widget。ComfyUI 在序列化图表发送给后端时，会读取这个 widget 的 `value`。如果上传还没完成，或者赋值没生效，后端收到的就是空字符串。

2.  **ComfyUI 原生 MaskEditor 机制**：
    *   ComfyUI 的 `MaskEditor` 是通过 `LoadImage` 节点的扩展实现的。
    *   它会将 Mask 数据保存为图片，并把图片名称填入 `image` widget。
    *   它利用了 `upload/image` 接口。

## 解决方案

我们需要确保：**Mask 图片必须先上传成功，并且文件名必须正确写入 Widget，然后用户点击运行（或自动触发运行）时，后端才能读到。**

### 1. 改进前端上传逻辑 (js/crop_ui.js)
*   **确保同步等待**：上传过程必须是 `await` 的，确保在关闭编辑器之前，文件名已经拿到了。
*   **Widget 赋值**：`this.w_mask_path.value = filename;` 是正确的，但为了保险，我们可以打印日志确认。
*   **触发图表更新**：赋值后，调用 `app.graph.setDirtyCanvas(true, true)` 确保 ComfyUI 知道图表变了。

### 2. 后端读取逻辑 (nodes.py)
*   **增加调试日志**：在 `crop` 函数里打印 `brush_mask_path` 的值，看看后端到底收到了什么。如果是空字符串，说明前端传值失败。
*   **目录问题**：ComfyUI 的 `upload/image` 默认上传到 `input` 目录。我们后端读取时也是从 `input` 目录读。这应该没问题。

### 3. 参考 ComfyUI 原生实现（LoadImage）
*   原生 MaskEditor 是把 mask 存成原本 image 的 alpha 通道或者单独的 mask 文件。
*   我们现在的做法是存单独文件 `brush_mask_{timestamp}.png`，这是合理的。

### 修正计划
1.  **调试优先**：修改 `nodes.py`，加入 print 语句，打印接收到的 `brush_mask_path`。
2.  **前端强化**：
    *   在上传成功后，显式调用 `this.w_mask_path.callback(filename)`（如果有回调的话）。
    *   或者直接更新 `widgets_values`。
    *   **关键**：我怀疑 `this.w_mask_path` 可能没被正确找到，或者因为它是 hidden 的，某些行为不同。我们将检查 `this.widgets` 列表。
3.  **上传参数修正**：
    *   原生上传接口返回的 JSON 里包含 `name` 字段（可能被重命名了）。
    *   我们必须使用**接口返回的文件名**，而不是我们自己生成的。ComfyUI 可能会自动重命名文件（例如 `brush_mask_123.png` 变成 `brush_mask_123_001.png`）。
    *   **这是一个巨大的坑**：如果我们忽略了返回结果，直接用自己生成的名字，就会导致后端找不到文件。

### 详细步骤
1.  修改 `js/crop_ui.js`：
    *   在 `api.fetchApi` 的响应中解析 JSON。
    *   获取 `response.name`，这才是服务器上真正的文件名。
    *   将这个真实文件名填入 `brush_mask_path`。
2.  修改 `nodes.py`：
    *   加入 print 调试。
    *   确保从 `input` 目录读取。

请确认执行。