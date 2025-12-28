import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "Comfy.RatioCropNode",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "RatioCropNode") {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            this.img = null;         // 原始图片对象
            this.previewImg = null;  // 带框的预览图对象
            this.maskImg = null;     // 用户绘制的蒙版对象 (HTMLImageElement)
            
            // 获取 Widgets
            this.w_image = this.widgets.find(w => w.name === "image");
            this.w_ratio = this.widgets.find(w => w.name === "ratio");
            this.w_x = this.widgets.find(w => w.name === "crop_x");
            this.w_y = this.widgets.find(w => w.name === "crop_y");
            this.w_w = this.widgets.find(w => w.name === "crop_w");
            this.w_h = this.widgets.find(w => w.name === "crop_h");
            this.w_mask_path = this.widgets.find(w => w.name === "brush_mask_path");
            this.w_padding = this.widgets.find(w => w.name === "padding");
            
            // 关键：在前端隐藏 brush_mask_path，使其不可见但功能正常
            if (this.w_mask_path) {
                this.w_mask_path.type = "hidden";
                this.w_mask_path.computeSize = () => [0, -4]; // 负高度以完全隐藏
            }

            // 添加按钮
            this.cropBtn = this.addWidget("button", "选定裁切范围", null, () => {
                // 如果按钮被禁用（通过修改 label 或 style），则不执行
                // 检查 image_input 是否连接
                if (this.inputs) {
                    const imageInput = this.inputs.find(i => i.name === "image_input");
                    if (imageInput && imageInput.link !== null) {
                        alert("已连接外部图片输入，请直接运行节点，无需手动裁切。");
                        return;
                    }
                }

                // 在点击按钮时，再次尝试获取最新的 image widget 值并加载
                // 这是一个双重保险，防止 callback 没触发或者加载失败
                const currentImageName = this.w_image.value;
                if (currentImageName && (!this.img || this.img.name !== currentImageName)) {
                     // 尝试同步加载（虽然 loadImage 是异步的，但我们可以在这里触发它）
                     this.loadImage(currentImageName);
                     // 由于是异步，可能第一次点会提示未加载，但这能触发加载
                }

                if (this.img) {
                    this.showCropEditor();
                } else {
                    // 如果还没加载好，给个提示，但同时尝试加载
                    if (currentImageName) {
                         this.loadImage(currentImageName);
                         // 延迟一下再试，或者提示用户稍后
                         setTimeout(() => {
                             if (this.img) this.showCropEditor();
                             else alert("正在加载图片，请稍后再试...");
                         }, 500);
                    } else {
                        alert("请先上传或选择图片！");
                    }
                }
            });

            // 监听 image 变化
            const originalCallback = this.w_image.callback;
            this.w_image.callback = (value) => {
                originalCallback?.(value);
                this.loadImage(value);
            };

            // 监听连接变化，更新按钮状态
            const onConnectionsChange = nodeType.prototype.onConnectionsChange;
            this.onConnectionsChange = function(type, index, connected, link_info, slot_info) {
                if (onConnectionsChange) onConnectionsChange.apply(this, arguments);
                
                // 检查 image_input 连接状态
                // slot_info 包含 name, type 等
                // 如果没有 slot_info (有时发生)，我们需要遍历 inputs
                
                let isImageInputConnected = false;
                if (this.inputs) {
                    const imageInput = this.inputs.find(i => i.name === "image_input");
                    if (imageInput && imageInput.link !== null) {
                        isImageInputConnected = true;
                    }
                }
                
                if (this.cropBtn) {
                    if (isImageInputConnected) {
                        this.cropBtn.name = "🚫 使用外部输入中";
                        // ComfyUI 的 button widget 没有直接的 disabled 属性，我们通过回调拦截和改名来实现
                    } else {
                        this.cropBtn.name = "选定裁切范围";
                    }
                    this.setDirtyCanvas(true); // 刷新 UI 显示
                }
            };
            
            // 初始化时检查一次
            setTimeout(() => {
                 if (this.onConnectionsChange) this.onConnectionsChange();
            }, 100);

            // 修正：对于粘贴图片，ComfyUI 可能不会触发 callback，或者值传递不完整
            // 我们需要 hook 节点的 onInputAdded 或者 check 变化
            // 但最直接的是重写 onNodeCreated 里的逻辑，确保加载
            
            // 增加一个 periodic check (可选) 或者依赖 ComfyUI 的 graph update
            
            // 强制加载一次初始值
            if (this.w_image.value) {
                this.loadImage(this.w_image.value);
            }
            
            // 监听粘贴事件 (paste) - ComfyUI 全局处理了 paste，会生成节点或更新 widget
            // 当 widget 值变化时，上面的 callback 会被调用。
            // 但是，对于 paste 的图片，value 可能是 "pasted/image.png"
            // 我们在 loadImage 里已经处理了 pasted/ 前缀的路径查找。
            // 问题可能在于：点击“选定裁切范围”时，this.img 还没更新？
            
            return r;
        };

        // 新增：监听执行完成事件，用于更新预览图 (特别是当使用 image_input 时)
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function(message) {
            onExecuted?.apply(this, arguments);

            // 检查是否有 UI 图像返回 (我们在 Python 端返回了预览图)
            if (message && message.ui && message.ui.images) {
                const imgs = message.ui.images;
                if (imgs.length > 0) {
                    const imgData = imgs[0];
                    // 构建预览图 URL
                    const url = api.apiURL(`/view?filename=${encodeURIComponent(imgData.filename)}&type=${imgData.type}&subfolder=${encodeURIComponent(imgData.subfolder)}`);
                    
                    // 加载并显示
                    const newPreview = new Image();
                    newPreview.onload = () => {
                        this.previewImg = newPreview;
                        // 更新 ComfyUI 的默认缩略图
                        if (this.imgs) {
                            this.imgs[0] = newPreview;
                        } else {
                            this.imgs = [newPreview];
                        }
                        
                        // 强制刷新节点显示
                        this.setDirtyCanvas(true, true);
                    };
                    newPreview.src = url;
                }
            }
        };

        nodeType.prototype.loadImage = function(imageName) {
            if (!imageName) {
                this.img = null;
                this.previewImg = null;
                this.maskImg = null;
                this.imgs = null;
                this.setSize([this.size[0], 220]);
                this.setDirtyCanvas(true, true);
                return;
            }

                const tryLoad = (name, type) => {
                return new Promise((resolve, reject) => {
                    const img = new Image();
                    img.crossOrigin = "Anonymous"; // 允许跨域，防止污染
                    img.onload = () => resolve(img);
                    img.onerror = reject;
                    img.src = api.apiURL(`/view?filename=${encodeURIComponent(name)}&type=${type}`);
                });
            };

            const tryLoadWithSubfolder = (filename, subfolder, type) => {
                 return new Promise((resolve, reject) => {
                    const img = new Image();
                    img.crossOrigin = "Anonymous";
                    img.onload = () => resolve(img);
                    img.onerror = reject;
                    let url = `/view?filename=${encodeURIComponent(filename)}&type=${type}`;
                    if (subfolder) url += `&subfolder=${encodeURIComponent(subfolder)}`;
                    img.src = api.apiURL(url);
                });
            };

            // 智能尝试逻辑：
            // 1. 拆分路径，提取 subfolder
            // 2. 尝试全排列
            
            const splitPath = (path) => {
                const parts = path.split(/[/\\]/); // split by / or \
                if (parts.length > 1) {
                    const filename = parts.pop();
                    const subfolder = parts.join("/");
                    return { filename, subfolder };
                }
                return { filename: path, subfolder: "" };
            };

            const attemptLoad = async () => {
                const { filename, subfolder } = splitPath(imageName);
                const types = ['input', 'temp', 'output'];
                
                // 队列设计：
                // 1. 如果有 subfolder，优先尝试带 subfolder 的请求
                // 2. 尝试把整个 imageName 当作 filename 的请求 (兼容旧逻辑)
                
                for (const type of types) {
                    // 尝试 1: 分离 subfolder
                    if (subfolder) {
                        try {
                            const img = await tryLoadWithSubfolder(filename, subfolder, type);
                            this.img = img;
                            this.img._comfy_filename = imageName; // 标记文件名，用于一致性检查
                            this.w_w.value = 0; this.w_h.value = 0; this.maskImg = null;
                            this.updatePreview();
                            return;
                        } catch(e) {}
                    }
                    
                    // 尝试 2: 原始路径作为 filename
                    try {
                        const img = await tryLoad(imageName, type);
                        this.img = img;
                        this.img._comfy_filename = imageName; // 标记文件名
                        this.w_w.value = 0; this.w_h.value = 0; this.maskImg = null;
                        this.updatePreview();
                        return;
                    } catch(e) {}
                }
                
                console.error(`[RatioCropNode] Failed to load image: ${imageName}`);
                // 如果是手动点击触发的，可能需要给个反馈，但这里是通用逻辑
            };

            return attemptLoad();
        };

        // 核心：生成带框的静态预览图
        nodeType.prototype.updatePreview = function() {
            // 如果已经有了来自后端的预览图 (onExecuted 设置的)，优先显示它？
            // 不，通常 updatePreview 是在 UI 交互时调用的。如果用户调整了 widget，应该显示前端合成的预览。
            // 但是，如果用户使用的是 image_input，前端没有 this.img，所以下面的逻辑会直接 return。
            // 因此，onExecuted 设置的 this.imgs[0] 依然有效，不会被这里覆盖。
            
            if (!this.img) return;

            const canvas = document.createElement("canvas");
            canvas.width = this.img.width;
            canvas.height = this.img.height;
            const ctx = canvas.getContext("2d");

            // 1. 画原图
            ctx.drawImage(this.img, 0, 0);
            
            // 1.5 画蒙版 (如果存在)
            if (this.maskImg) {
                ctx.drawImage(this.maskImg, 0, 0);
            }

            // 2. 画遮罩和绿框
            const x = this.w_x.value;
            const y = this.w_y.value;
            const w = this.w_w.value;
            const h = this.w_h.value;
            
            // 只有当宽高有效时才绘制遮罩和绿框
            if (w > 0 && h > 0) {
                // 半透明遮罩
                ctx.fillStyle = "rgba(0, 0, 0, 0.6)";
                ctx.fillRect(0, 0, canvas.width, y);
                ctx.fillRect(0, y + h, canvas.width, canvas.height - (y + h));
                ctx.fillRect(0, y, x, h);
                ctx.fillRect(x + w, y, canvas.width - (x + w), h);

                // 绿框
                ctx.strokeStyle = "#00FF00";
                ctx.lineWidth = Math.max(2, canvas.width / 200); 
                ctx.strokeRect(x, y, w, h);
                
                // 尺寸文字
                const fontSize = Math.max(12, canvas.width / 40);
                ctx.fillStyle = "#00FF00";
                ctx.font = `bold ${fontSize}px Arial`;
                ctx.fillText(`${w}x${h}`, x, y - fontSize/2);
            }

            // 3. 生成预览图对象
            const previewUrl = canvas.toDataURL("image/jpeg", 0.8);
            const previewImg = new Image();
            previewImg.onload = () => {
                this.previewImg = previewImg;
                if (this.imgs) {
                    this.imgs[0] = previewImg; 
                } else {
                    this.imgs = [previewImg];
                }
                
                // 调整节点大小
                const widgetAreaHeight = 200; 
                const minWidth = 300;
                const targetW = Math.max(this.size[0], minWidth);
                const scale = targetW / previewImg.width;
                const targetH = widgetAreaHeight + (previewImg.height * scale) + 20;
                
                this.setSize([targetW, targetH]);
                this.setDirtyCanvas(true, true);
            };
            previewImg.src = previewUrl;
        };
        
        nodeType.prototype.onMouseDown = function(e, localPos, canvas) {
            if (localPos[1] > 200) {
                if (this.img) {
                    this.showCropEditor();
                    return true;
                }
            }
        };

        // 弹出编辑器逻辑
        nodeType.prototype.showCropEditor = function() {
            const overlay = document.createElement("div");
            Object.assign(overlay.style, {
                position: "fixed", top: "0", left: "0", width: "100%", height: "100%",
                backgroundColor: "rgba(0,0,0,0.85)", zIndex: "10000", display: "flex",
                flexDirection: "column", alignItems: "center", justifyContent: "center"
            });

            // --- 工具栏 ---
            const toolbar = document.createElement("div");
            Object.assign(toolbar.style, {
                marginBottom: "10px", display: "flex", gap: "10px", color: "white", alignItems: "center"
            });
            
            // 比例选择 (新增 Smart)
            const ratioSelect = document.createElement("select");
            // Smart 放在第一位
            ["Smart", "1:1", "3:4", "4:3", "9:16", "16:9", "21:9"].forEach(r => {
                const opt = document.createElement("option");
                opt.value = r;
                opt.text = r;
                // 如果当前 ratio 不在列表里（比如是旧的），默认选 Smart
                // 但如果节点里存的是 "1:1"，就选 "1:1"
                if (r === this.w_ratio.value) opt.selected = true;
                ratioSelect.appendChild(opt);
            });
            // 默认选中 Smart
            if (!this.w_ratio.value || this.w_ratio.value === "Free") {
                ratioSelect.value = "Smart";
            }
            
            const clearMaskBtn = document.createElement("button");
            clearMaskBtn.innerText = "清除涂抹";
            
            // 工具：画笔 / 橡皮擦 / 油漆桶 / 框选
            let toolMode = "brush"; // brush, eraser, fill, box
            
            const brushBtn = document.createElement("button");
            brushBtn.innerText = "🖌️";
            brushBtn.title = "画笔";
            brushBtn.style.backgroundColor = "#666"; // Active color
            
            const eraserBtn = document.createElement("button");
            eraserBtn.innerText = "🧹";
            eraserBtn.title = "橡皮擦";
            eraserBtn.style.backgroundColor = "#333";

            const fillBtn = document.createElement("button");
            fillBtn.innerText = "🪣";
            fillBtn.title = "油漆桶 (填充)";
            fillBtn.style.backgroundColor = "#333";
            
            const undoBtn = document.createElement("button");
            undoBtn.innerText = "↩️";
            undoBtn.title = "撤销 (Ctrl+Z)";
            
            const updateToolBtnStyles = () => {
                brushBtn.style.backgroundColor = toolMode === "brush" ? "#666" : "#333";
                eraserBtn.style.backgroundColor = toolMode === "eraser" ? "#666" : "#333";
                fillBtn.style.backgroundColor = toolMode === "fill" ? "#666" : "#333";
            };

            brushBtn.onclick = () => { toolMode = "brush"; updateToolBtnStyles(); draw(); };
            eraserBtn.onclick = () => { toolMode = "eraser"; updateToolBtnStyles(); draw(); };
            fillBtn.onclick = () => { toolMode = "fill"; updateToolBtnStyles(); draw(); };

            // 历史记录栈
            const historyStack = [];
            const saveHistory = () => {
                if (historyStack.length > 20) historyStack.shift(); // 限制步数
                historyStack.push(maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height));
            };
            
            const undo = () => {
                if (historyStack.length > 0) {
                    const prevState = historyStack.pop();
                    maskCtx.putImageData(prevState, 0, 0);
                    draw();
                }
            };
            
            undoBtn.onclick = undo;
            
            // Ctrl+Z 撤销
            window.addEventListener("keydown", (e) => {
                if ((e.ctrlKey || e.metaKey) && e.key === "z") {
                    e.preventDefault();
                    undo();
                }
            });

            // 画笔大小
            const brushSizeInput = document.createElement("input");
            brushSizeInput.type = "range";
            brushSizeInput.min = "1";
            brushSizeInput.max = "200"; // 增加最大值
            brushSizeInput.value = "50"; // 增加默认值
            brushSizeInput.title = "画笔大小";
            
            const paddingLabel = document.createElement("span");
            paddingLabel.innerText = "冗余(%): 20";
            const paddingInput = document.createElement("input");
            paddingInput.type = "range";
            paddingInput.min = "0";
            paddingInput.max = "100";
            paddingInput.value = this.w_padding ? this.w_padding.value : "20"; 
            paddingInput.title = "冗余比例";
            
            // 实时更新数值显示
            paddingInput.oninput = () => {
                paddingLabel.innerText = `冗余(%): ${paddingInput.value}`;
            };

            const confirmBtn = document.createElement("button");
            confirmBtn.innerText = "确认裁切";
            confirmBtn.style.padding = "5px 15px";
            confirmBtn.style.cursor = "pointer";

            const cancelBtn = document.createElement("button");
            cancelBtn.innerText = "取消";
            cancelBtn.style.padding = "5px 15px";

            toolbar.appendChild(document.createTextNode("比例: "));
            toolbar.appendChild(ratioSelect);
            toolbar.appendChild(document.createTextNode(" | "));
            toolbar.appendChild(brushBtn);
            toolbar.appendChild(eraserBtn);
            toolbar.appendChild(fillBtn);
            toolbar.appendChild(brushSizeInput);
            toolbar.appendChild(clearMaskBtn);
            toolbar.appendChild(undoBtn);
            toolbar.appendChild(paddingLabel);
            toolbar.appendChild(paddingInput);
            toolbar.appendChild(confirmBtn);
            toolbar.appendChild(cancelBtn);
            overlay.appendChild(toolbar);

            // --- 画布容器 ---
            const canvasContainer = document.createElement("div");
            Object.assign(canvasContainer.style, {
                position: "relative", width: "80%", height: "80%", backgroundColor: "#333",
                overflow: "hidden", display: "flex", justifyContent: "center", alignItems: "center",
                cursor: "crosshair"
            });
            overlay.appendChild(canvasContainer);

            const canvas = document.createElement("canvas");
            canvasContainer.appendChild(canvas);

            document.body.appendChild(overlay);

            // --- 状态管理 ---
            const img = this.img;
            let scale = 1;
            let offsetX = 0, offsetY = 0;
            
            let crop = { 
                x: this.w_x.value, y: this.w_y.value, 
                w: this.w_w.value, h: this.w_h.value 
            };
            if (crop.w <= 0) { crop.w = 512; crop.h = 512; }

            const maskCanvas = document.createElement("canvas");
            maskCanvas.width = img.width;
            maskCanvas.height = img.height;
            const maskCtx = maskCanvas.getContext("2d");
            
            let isDrawing = false;
            let isMovingCrop = false;
            let startPos = { x: 0, y: 0 };
            let startCrop = { ...crop };

            const fitCanvas = () => {
                const rect = canvasContainer.getBoundingClientRect();
                canvas.width = rect.width;
                canvas.height = rect.height;
                const scaleW = canvas.width / img.width;
                const scaleH = canvas.height / img.height;
                scale = Math.min(scaleW, scaleH) * 0.9;
                offsetX = (canvas.width - img.width * scale) / 2;
                offsetY = (canvas.height - img.height * scale) / 2;
                draw();
            };

            const getImgPos = (e) => {
                const rect = canvas.getBoundingClientRect();
                const x = Math.round((e.clientX - rect.left - offsetX) / scale); // 改为 round
                const y = Math.round((e.clientY - rect.top - offsetY) / scale); // 改为 round
                // 边界限制，防止断触
                // 增加 0.5 的容错空间，防止正好压线导致 Math.floor 问题
                return {
                    x: Math.max(0, Math.min(img.width, x)),
                    y: Math.max(0, Math.min(img.height, y))
                };
            };

            // --- Flood Fill 算法 ---
            const floodFill = (startX, startY) => {
                const w = maskCanvas.width;
                const h = maskCanvas.height;
                
                // 检查起始点是否在画布内
                if (startX < 0 || startX >= w || startY < 0 || startY >= h) return;

                const imageData = maskCtx.getImageData(0, 0, w, h);
                const data = imageData.data; // Uint8ClampedArray [r, g, b, a, ...]
                
                // 目标颜色: 红色 (255, 0, 0, 128) -> alpha 约为 128
                // 我们实际上只关心 alpha。如果是透明的 (0)，就填成不透明。
                // 如果已经有颜色了，就不填。
                
                const getAlpha = (x, y) => data[(y * w + x) * 4 + 3];
                const setPixel = (x, y) => {
                    const idx = (y * w + x) * 4;
                    data[idx] = 255;     // R
                    data[idx + 1] = 0;   // G
                    data[idx + 2] = 0;   // B
                    data[idx + 3] = 255; // A (100% 不透明)
                };

                const startAlpha = getAlpha(startX, startY);
                if (startAlpha > 10) return; // 已经有颜色了，不重复填充

                // 使用栈进行迭代填充 (避免递归爆栈)
                const stack = [[startX, startY]];
                
                while (stack.length > 0) {
                    const [x, y] = stack.pop();
                    
                    if (x < 0 || x >= w || y < 0 || y >= h) continue;
                    if (getAlpha(x, y) > 10) continue; // 边界
                    
                    setPixel(x, y);
                    
                    stack.push([x + 1, y]);
                    stack.push([x - 1, y]);
                    stack.push([x, y + 1]);
                    stack.push([x, y - 1]);
                }
                
                maskCtx.putImageData(imageData, 0, 0);
            };

            // --- 自动计算裁切框 (包含智能比例) ---
            const autoCropFromMask = () => {
                const w = maskCanvas.width;
                const h = maskCanvas.height;
                const pixels = maskCtx.getImageData(0, 0, w, h).data;
                
                let minX = w, minY = h, maxX = 0, maxY = 0;
                let found = false;

                for (let y = 0; y < h; y++) {
                    for (let x = 0; x < w; x++) {
                        if (pixels[(y * w + x) * 4 + 3] > 0) {
                            if (x < minX) minX = x;
                            if (x > maxX) maxX = x;
                            if (y < minY) minY = y;
                            if (y > maxY) maxY = y;
                            found = true;
                        }
                    }
                }

                if (!found) return;

                // 计算百分比 padding
                let contentW = maxX - minX;
                let contentH = maxY - minY;
                // 使用长边计算基础 padding
                const baseSize = Math.max(contentW, contentH);
                const paddingPercent = parseInt(paddingInput.value) || 0;
                const padding = Math.round(baseSize * (paddingPercent / 100));

                minX = Math.max(0, minX - padding);
                minY = Math.max(0, minY - padding);
                maxX = Math.min(w, maxX + padding);
                maxY = Math.min(h, maxY + padding);

                let targetW = maxX - minX;
                let targetH = maxY - minY;
                
                // --- 智能比例匹配 ---
                let ratioStr = ratioSelect.value;
                
                if (ratioStr === "Smart") {
                    const currentRatio = targetW / targetH;
                    // 定义预设比例
                    const ratios = [
                        { name: "1:1", val: 1.0 },
                        { name: "3:4", val: 3/4 },
                        { name: "4:3", val: 4/3 },
                        { name: "9:16", val: 9/16 },
                        { name: "16:9", val: 16/9 },
                        { name: "21:9", val: 21/9 }
                    ];
                    
                    // 找最近邻
                    let bestR = ratios[0];
                    let minDiff = Math.abs(currentRatio - bestR.val);
                    
                    for (let i = 1; i < ratios.length; i++) {
                        const diff = Math.abs(currentRatio - ratios[i].val);
                        if (diff < minDiff) {
                            minDiff = diff;
                            bestR = ratios[i];
                        }
                    }
                    
                    // 自动切换下拉菜单
                    // ratioSelect.value = bestR.name; // <--- 移除这行，保持 Smart 选中状态
                    ratioStr = bestR.name; // 更新当前计算用的 ratio
                }

                if (ratioStr !== "Free" && ratioStr !== "Smart") {
                    let r = 1;
                    if (ratioStr === "1:1") r = 1;
                    else if (ratioStr === "3:4") r = 3/4;
                    else if (ratioStr === "4:3") r = 4/3;
                    else if (ratioStr === "9:16") r = 9/16;
                    else if (ratioStr === "16:9") r = 16/9;
                    else if (ratioStr === "21:9") r = 21/9;

                    const currentR = targetW / targetH;
                    if (currentR < r) {
                        const newW = targetH * r;
                        const diff = newW - targetW;
                        minX -= diff / 2;
                        targetW = newW;
                    } else {
                        const newH = targetW / r;
                        const diff = newH - targetH;
                        minY -= diff / 2;
                        targetH = newH;
                    }
                }

                if (minX < 0) minX = 0;
                if (minY < 0) minY = 0;
                if (minX + targetW > w) minX = w - targetW;
                if (minY + targetH > h) minY = h - targetH;
                if (minX < 0) minX = 0;
                if (minY < 0) minY = 0;
                if (minX + targetW > w) targetW = w - minX;
                if (minY + targetH > h) targetH = h - minY;

                crop.x = minX;
                crop.y = minY;
                crop.w = targetW;
                crop.h = targetH;
            };

            const draw = () => {
                const ctx = canvas.getContext("2d");
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.save();
                ctx.translate(offsetX, offsetY);
                ctx.scale(scale, scale);
                ctx.drawImage(img, 0, 0);
                
                // 绘制 maskCanvas，使用全局半透明
                ctx.save();
                ctx.globalAlpha = 0.5; // 统一半透明度
                ctx.drawImage(maskCanvas, 0, 0);
                ctx.restore();
                
                ctx.restore();
            };

            const isPointInCrop = (p) => {
                return p.x >= crop.x && p.x <= crop.x + crop.w &&
                       p.y >= crop.y && p.y <= crop.y + crop.h;
            };

            // --- 画布操作逻辑 ---
            let isSpacePressed = false;
            let isPanning = false;
            let startPan = { x: 0, y: 0 };

            window.addEventListener("keydown", (e) => {
                if (e.code === "Space" && !isSpacePressed) {
                    isSpacePressed = true;
                    canvasContainer.style.cursor = "grab";
                }
            });

            window.addEventListener("keyup", (e) => {
                if (e.code === "Space") {
                    isSpacePressed = false;
                    isPanning = false;
                    canvasContainer.style.cursor = "crosshair";
                }
            });

            canvas.addEventListener("wheel", (e) => {
                e.preventDefault();
                const zoomSpeed = 0.1;
                const factor = e.deltaY > 0 ? (1 - zoomSpeed) : (1 + zoomSpeed);
                
                // 限制缩放范围
                const newScale = scale * factor;
                if (newScale < 0.1 || newScale > 10) return;

                const rect = canvas.getBoundingClientRect();
                const mouseX = e.clientX - rect.left;
                const mouseY = e.clientY - rect.top;

                // 计算缩放中心
                offsetX = mouseX - (mouseX - offsetX) * factor;
                offsetY = mouseY - (mouseY - offsetY) * factor;
                scale = newScale;
                
                draw();
            });

            canvas.addEventListener("mousedown", (e) => {
                // 平移模式：空格键+左键 或 中键
                if (isSpacePressed || e.button === 1) {
                    isPanning = true;
                    startPan = { x: e.clientX, y: e.clientY };
                    canvasContainer.style.cursor = "grabbing";
                    return;
                }

                const pos = getImgPos(e);
                
                const edgeDist = 10 / scale;
                const onLeft = Math.abs(pos.x - crop.x) < edgeDist;
                const onRight = Math.abs(pos.x - (crop.x + crop.w)) < edgeDist;
                const onTop = Math.abs(pos.y - crop.y) < edgeDist;
                const onBottom = Math.abs(pos.y - (crop.y + crop.h)) < edgeDist;
                const inside = isPointInCrop(pos);

                // 其他模式下 (Brush/Eraser/Fill)
                if (toolMode === "fill") {
                        saveHistory(); // 填充前保存历史
                        floodFill(pos.x, pos.y);
                        draw();
                } else {
                        saveHistory(); // 绘制前保存历史
                        isDrawing = true;
                        maskCtx.beginPath();
                        maskCtx.lineCap = "round";
                        maskCtx.lineJoin = "round";
                        maskCtx.lineWidth = parseInt(brushSizeInput.value);
                        
                        if (toolMode === "eraser") {
                            maskCtx.globalCompositeOperation = "destination-out";
                            maskCtx.strokeStyle = "rgba(0, 0, 0, 1)";
                        } else {
                            maskCtx.globalCompositeOperation = "source-over";
                            maskCtx.strokeStyle = "rgba(255, 0, 0, 1)"; // 纯红色，不透明
                        }
                        
                        maskCtx.moveTo(pos.x, pos.y);
                        maskCtx.lineTo(pos.x, pos.y);
                        maskCtx.stroke();
                        draw();
                }
            });

            // 修改事件监听目标为 canvas 容器或 canvas 本身，以提高响应性
            // 同时保留 window 上的 mouseup 以防拖出界外松开
            
            canvas.addEventListener("mousemove", (e) => {
                if (isPanning) {
                    const dx = e.clientX - startPan.x;
                    const dy = e.clientY - startPan.y;
                    offsetX += dx;
                    offsetY += dy;
                    startPan = { x: e.clientX, y: e.clientY };
                    draw();
                    return;
                }

                const pos = getImgPos(e);
                if (isDrawing) {
                    maskCtx.lineTo(pos.x, pos.y);
                    maskCtx.stroke();
                    draw();
                }
            });

            window.addEventListener("mouseup", () => {
                if (isPanning) {
                    isPanning = false;
                    if (isSpacePressed) canvasContainer.style.cursor = "grab";
                    else canvasContainer.style.cursor = "crosshair";
                }
                if (isDrawing) {
                    isDrawing = false;
                    maskCtx.closePath();
                    // 移除自动计算: autoCropFromMask();
                    draw();
                }
                isMovingCrop = false;
            });

            clearMaskBtn.onclick = () => {
                maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
                draw();
            };

            confirmBtn.onclick = async () => {
                // 点击确认时，如果不是框选模式，才计算 Mask 对应的框
                if (toolMode !== "box") {
                    autoCropFromMask();
                } else {
                    // 如果是框选模式，我们需要把当前的框，转换成 mask，以便后端逻辑统一
                    // 清空蒙版
                    maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
                    // 填充红色矩形
                    maskCtx.fillStyle = "rgba(255, 0, 0, 1)";
                    maskCtx.fillRect(crop.x, crop.y, crop.w, crop.h);
                }
                
                // --- 上传 Mask ---
                try {
                    // 将 Mask Canvas 转换为 Blob
                    const blob = await new Promise(resolve => maskCanvas.toBlob(resolve, 'image/png'));
                    if (blob) {
                        const formData = new FormData();
                        // 生成唯一文件名
                        const filename = `brush_mask_${Date.now()}.png`;
                        formData.append('image', blob, filename);
                        formData.append('overwrite', 'true');
                        formData.append('type', 'input'); // 确保类型为 input

                        const resp = await api.fetchApi("/upload/image", {
                            method: "POST",
                            body: formData
                        });

                        if (resp.status !== 200) {
                            throw new Error(`Upload failed with status ${resp.status}: ${resp.statusText}`);
                        }

                        const result = await resp.json();
                        // 关键修正：使用服务器返回的真实文件名
                        const serverFilename = result.name;

                        // 回填文件名到隐藏 Widget
                        if (this.w_mask_path) {
                            this.w_mask_path.value = serverFilename;
                            console.log("[RatioCropNode] Mask uploaded:", serverFilename);
                            // 强制触发更新，确保 ComfyUI 知道图表已变更
                            app.graph.setDirtyCanvas(true, true);
                        }
                    }
                } catch (e) {
                    console.error("[RatioCropNode] Mask upload failed:", e);
                    alert(`蒙版上传失败: ${e.message}\n请检查控制台日志。`);
                }

                this.w_x.value = Math.round(crop.x);
                this.w_y.value = Math.round(crop.y);
                this.w_w.value = Math.round(crop.w);
                this.w_h.value = Math.round(crop.h);
                this.w_ratio.value = ratioSelect.value; 
                if (this.w_padding) this.w_padding.value = parseInt(paddingInput.value);
                
                // 保存当前的 maskCanvas 内容到 this.maskImg 以便 updatePreview 使用
                const maskImg = new Image();
                maskImg.src = maskCanvas.toDataURL();
                maskImg.onload = () => {
                    this.maskImg = maskImg;
                    this.updatePreview();
                };
                
                document.body.removeChild(overlay);
            };

            cancelBtn.onclick = () => {
                document.body.removeChild(overlay);
            };

            fitCanvas();
            window.addEventListener("resize", fitCanvas);
        };
    }
});
