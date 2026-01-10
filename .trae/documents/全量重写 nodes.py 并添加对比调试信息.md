# 终极分析：为什么数字变了还是对不上？

用户提供了新的截图：
1.  **UI 面板**：`crop_w: 676`, `crop_h: 1202`。
2.  **中间预览图**：`684 x 1216`。
3.  **绿框文字**：`676 x 1202` (与 UI 面板一致，但与中间图不一致)。

**数据分析**：
-   **中间图 (684 x 1216)**：
    -   `684 / 1216 = 0.5625` (精确 9:16)。
    -   `1216 / 16 = 76`。 `76 * 9 = 684`。
    -   **这是经过整数量化后的完美结果。**

-   **绿框 & UI (676 x 1202)**：
    -   `676 / 1202 = 0.562396` (近似 9:16，但不是精确的)。
    -   `1202 / 16 = 75.125`。
    -   **这是没有经过整数量化的结果！**

**结论：**
后端计算确实生成了完美的 `684x1216` 并输出了图片。
但是！前端 UI 和绿框预览图依然显示的是**修正前**（或修正不彻底）的 `676x1202`。

**为什么绿框是错的？**
我在之前的代码里修改了：
```python
            if image_input.shape[0] == 1:
                 bx, by, bw, bh = final_x, final_y, final_w, final_h
```
如果绿框显示 `676`，说明 `final_w` 是 `676`。
如果 `final_w` 是 `676`，说明 `apply_ratio_logic` 返回了 `676`。
如果 `apply_ratio_logic` 返回了 `676`，那说明整数量化没生效？
**但是！** 中间图是 `684`。
中间图也是用 `final_w` 生成的啊！
```python
                res = core_crop_logic(..., final_w, ...)
```
如果 `final_w` 是 676，那 `core_crop_logic` 接收到的就是 676。
`core_crop_logic` 内部再次调用 `apply_ratio_logic`，把它变成了 684。
然后裁切出 684 的图。
返回 684 的数据。

**破案了！**
**`RatioCropNode.crop` 里的 `final_w` 依然是 676（未完全修正），而 `core_crop_logic` 里的 `apply_ratio_logic` 把它修正成了 684。**
这意味着：**`RatioCropNode.crop` 里的 `apply_ratio_logic` 和 `core_crop_logic` 里的 `apply_ratio_logic` 行为不一致！**
或者，`RatioCropNode.crop` 里的 `apply_ratio_logic` 根本没执行？

看代码：
```python
                # --- Force Apply Ratio Logic ---
                if ratio != "Free":
                    final_x, final_y, final_w, final_h = apply_ratio_logic(...)
```
如果 `ratio` 是 "Smart"。
`676 / 1202 = 0.5623`。
Smart 应该检测为 9:16。
应该算出 684。

**为什么算出 676？**
除非... `apply_ratio_logic` 的实现有问题？
在 `apply_ratio_logic` 中：
```python
        k = int(np.ceil(k))
```
`1202 / 16 = 75.125`。
如果基于高度，`k = 75.125`。 `ceil` -> 76。
`76 * 16 = 1216`。
`76 * 9 = 684`。
**怎么算都是 684。**

**除非！`RatioCropNode.crop` 里没有引入 `numpy`？**
不，文件头引入了。

**除非！`RatioCropNode.crop` 里的 `ratio` 变量不是 "Smart"？**
但 UI 显示 Smart。

**真正的嫌疑人：`padding`。**
截图显示 `padding: 2`。
`use_mask_bbox` 可能生效了？
如果生效了，`final_w` 会被重置为 Mask BBox + Padding。
如果 Mask BBox + Padding 刚好是 676。
然后 `apply_ratio_logic` 把它变成了 684。
**如果 `RatioCropNode.crop` 里的 `apply_ratio_logic` 没生效**，那 `final_w` 就是 676。
**为什么没生效？**
**因为我之前的 SearchReplace 可能失败了！**
虽然我 cat 出来看是对的。
但这种逻辑矛盾太强烈了。

**还有一个微小的可能性：**
用户截图里的绿框文字没有显示 `(Smart)`。
**我看不到 `(Smart)`！**
**我上次修改代码增加了 `if ratio != "Free": text_str += f" ({ratio})"`。**
**如果绿框上没有 `(Smart)`，那说明 `ratio` 等于 "Free"！**
**如果 `ratio` 是 "Free"，那 `apply_ratio_logic` 直接返回原值（676）！**
**所以 `final_w` 是 676！**
**所以绿框是 676！**
**所以 UI 是 676！**

**但是！为什么中间图是 684？**
如果 `ratio` 是 "Free"，`core_crop_logic` 也应该收到 "Free"。
`core_crop_logic` 里的 `apply_ratio_logic` 也会跳过。
裁切出来的应该是 676。
**为什么裁切出来是 684？**
这说明 `core_crop_logic` 里的 `ratio` **不是** "Free"！

**结论：`RatioCropNode.crop` 里的 `ratio` 变量是 "Free"，而 `core_crop_logic` 里的 `ratio` 变量是 "Smart"？**
这怎么可能？
调用代码：
```python
res = core_crop_logic(..., ratio=ratio)
```
它们是同一个变量。

**等等！截图里的绿框文字真的没有 `(Smart)` 吗？**
我看不太清，但似乎只有 `676x1202`。
如果真的没有，那 `ratio` 就是 `Free`。
**如果 `ratio` 是 `Free`，那一切都解释通了（绿框 676）。**
**唯一解释不通的是中间图 684。**
中间图的标题是 `#248`。
有没有可能 `#248` 节点自己有某种魔法？
或者，**用户截图里的中间图不是本次运行的结果？**
不，分辨率变了（684x1216）。

**让我们再看一眼 `core_crop_logic` 的代码。**
```python
    # 3. 修正 Ratio String
    ratio_str = ratio
    if ratio == "Smart":
        ...
        ratio_str = best_r
```
这里修改了 `ratio_str`，但没有修改 `ratio` 参数。
裁切逻辑用的是 `ratio` 参数。

**假设：用户的 ComfyUI 里的 `ratio` 确实是 "Smart"。**
**那为什么绿框没显示？**
也许文字超出了图片范围被切掉了？
或者颜色太淡看不清？
或者我的代码没更新成功？

**无论如何，为了解决“对不上”的问题，我必须确保 `RatioCropNode` 里的 `final_w` 和 `core_crop_logic` 里的逻辑完全一致。**

**我现在的计划：**
1.  **强制转换 ratio**：在 `RatioCropNode.crop` 入口处，确保 `ratio` 是字符串且去除了空格。
2.  **强制打印调试日志到控制台**：虽然用户看不到，但我可以在代码里写，以防万一。
3.  **最关键的修复**：
    我怀疑 `apply_ratio_logic` 在 `RatioCropNode` 里虽然被调用了，但返回值没有正确赋给 `final_w`？
    ```python
    final_x, final_y, final_w, final_h = apply_ratio_logic(...)
    ```
    看起来是对的。

**有没有可能 `image_input.shape[0] != 1`？**
如果 `batch > 1`。
`bx, by, bw, bh = batch_results[0]...`
这时取的是 `core_crop_logic` 的返回值。
如果 `core_crop_logic` 返回的是 684。
那绿框就该是 684。
**现在绿框是 676。说明 `core_crop_logic` 返回的是 676？**
如果返回 676，那裁切图也该是 676。
**现在裁切图是 684。**
**这依然是那个死循环逻辑矛盾。**

**除非... `batch_results[0][7]` (w) 和 `image_tensor` 的宽度不一致。**
在 `core_crop_logic` 里：
```python
    w = int(round(crop_w)) # 684
    image.crop(..., w, ...) # 684
    crop_data = { "crop_w": w } # 684
    return (..., w, ...) # 684
```
完全一致。

**我放弃逻辑推演了。这一定是某种我看不到的低级错误（比如拼写错误，或者文件没保存）。**
**我将执行一次全量的 `nodes.py` 写入。**
我不使用 SearchReplace，而是把整个文件重写一遍。
确保每一个字符都是我期望的。
并且，我会在绿框上画一个**巨大的红色矩形**作为调试标记，确认代码生效了。

**等等，截图里的 `crop_y` 是 156。**
**UI 显示 `crop_y: 156`。**
**绿框高度 1202。**
**156 + 1202 = 1358。**
**原图高度 1536。**
**没有卡边缘。**
所以“卡边缘”理论不成立。

**总结**：
-   绿框是 676x1202 (9:16 approx)。
-   中间图是 684x1216 (9:16 exact)。
-   说明整数量化逻辑在裁切时生效了，但在预览/UI更新时没生效。
-   这说明 `RatioCropNode` 里的 `final_w` 计算逻辑有误，或者被覆盖了。

**我将在重写代码时，把 `final_w` 的计算逻辑写得再显式一点。**
并且，**我会把 `core_crop_logic` 返回的 `w` 强制覆盖 `final_w`（如果是单图模式）。**
这样，预览图就不得不和裁切结果一致了。

**执行方案：**
重写 `nodes.py`。
在 `RatioCropNode.crop` 的预览部分：
```python
            if image_input.shape[0] == 1:
                 # 相信 core_crop_logic 的返回值，因为它对应了真实的 image_tensor
                 bx, by, bw, bh = batch_results[0][5], batch_results[0][6], batch_results[0][7], batch_results[0][8]
                 
                 # 同时也更新 UI 更新列表
                 ui_update_values = [..., int(bw), int(bh)]
```
**之前的代码里，我是反过来的：**
```python
            if image_input.shape[0] == 1:
                 bx, by, bw, bh = final_x, final_y, final_w, final_h
```
我当时是为了修复“绿框宽了”的问题（假设 `final_w` 是对的，`batch_results` 是错的）。
现在看来，`final_w` 是错的（676），`batch_results` 是对的（应该是 684，因为它对应裁切图）。
**等等，如果 `batch_results` 是 684，那之前的 bug（绿框 995）是怎么回事？**
当时绿框是 995，裁切是 864。
说明当时 `batch_results` 是 995。
说明当时 `core_crop_logic` 返回了 995，但裁切了 864。
**这依然是那个死结。**

**不管了，我把两头都堵死。**
1.  `RatioCropNode` 里算一遍 684。
2.  `core_crop_logic` 里算一遍 684。
3.  预览用 `batch_results`（理论上是 684）。
4.  如果有差异，我在绿框上把两个值都打印出来：`Final: {final_w} | Batch: {bw}`。

