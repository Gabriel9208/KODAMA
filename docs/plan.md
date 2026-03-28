# KODAMA Project Plan v2 — 方案 C：中間路線

## 你的背景（校正後）

- C++ / OpenGL：理解 GPU 記憶體管理、矩陣變換、rendering pipeline
- PyTorch：實作過 U-Net / ResNet34-UNet binary segmentation（完整訓練流程）
- 已完成：SANPO 資料前處理 pipeline、Dataset class、PQ metric
- 理論：李宏毅 ML + CS231n
- 短板：尚未處理過 multi-class + instance-level 的訓練，ML debug 經驗有限

---

## 核心策略：凍結 Backbone，自己設計 Panoptic Decoder

不套 API 當黑盒，也不重寫整個 YOLO head。
取中間路線 — 借用成熟的 instance segmentation 能力，自己設計並訓練 semantic decoder。

```
┌─────────────────────────────────────────────────────────┐
│  YOLO11n-seg Backbone + Neck (凍結 or 微調最後幾層)       │
│  ┌──────────────────┐                                    │
│  │  FPN Feature Maps │──┬── P2 ── P3 ── P4 ── P5        │
│  └──────────────────┘  │                                 │
│                         │                                 │
│    ┌────────────────────┴────────────────────┐            │
│    │                                         │            │
│    ▼                                         ▼            │
│  [原本的 Detection + Seg Head]    [你自己寫的 Semantic     │
│   (不動，fine-tune only)           Decoder]               │
│    │                                         │            │
│    ▼                                         ▼            │
│  Instance Masks (Things)          Stuff Segmentation Map  │
│  bbox, class, mask coefficients   人行道/道路/草地/天空...  │
│    │                                         │            │
│    └──────────────┬──────────────────────────┘            │
│                   ▼                                       │
│         [Panoptic Fusion Module] ← 你自己寫的              │
│                   │                                       │
│                   ▼                                       │
│         [Depth-aware Risk Scoring] ← 你自己寫的            │
│          結合 SANPO depth map                              │
│                   │                                       │
│                   ▼                                       │
│         Panoptic Map + Risk Level per Object              │
└─────────────────────────────────────────────────────────┘
```

### 為什麼這個方案有足夠的原創性

| 你寫的模組             | 技術內容                                                                   | 面試怎麼講                                                                                                                                  |
| ---------------------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Semantic Decoder       | 自己設計的輕量網路：從 FPN feature 接出，做 multi-class stuff segmentation | 「我設計了一個 lightweight decoder，用 depthwise separable conv + bilinear upsample 從共享 backbone 的 P3 feature 產生 stuff segmentation」 |
| Panoptic Fusion        | 自己寫的衝突解決演算法：instance mask 與 stuff 像素重疊時的決策邏輯        | 「我實作了 confidence-based panoptic fusion，處理 things/stuff 的 label conflict」                                                          |
| Depth Risk Scoring     | 結合 SANPO depth map 的風險量化系統                                        | 「我整合 depth information 做距離感知的 risk assessment，2m 內障礙物 recall > 95%」                                                         |
| 完整 Training Pipeline | 兩階段訓練 + loss balancing                                                | 「先訓練 instance seg，再凍結 backbone 訓練 semantic decoder，避免多 loss 干擾」                                                            |
| PQ Evaluation          | 從零實作 Panoptic Quality 指標                                             | 「我自己實作了 PQ metric 而不是呼叫現成 library」                                                                                           |

### 為什麼風險可控

- Detection head 完全不動 → instance segmentation 一定能跑
- Semantic decoder 獨立訓練 → 不會影響 instance 的收斂
- 退路明確：最差情況砍掉 decoder，退回 fine-tune only，手上還是有完整結果

### 為什麼選 YOLO11 而不是 YOLOv8 / YOLOv12 / YOLO26

| 模型         | 參數量 (nano-seg) | 選或不選 | 原因                                                                                                                             |
| ------------ | ----------------- | -------- | -------------------------------------------------------------------------------------------------------------------------------- |
| YOLOv8n-seg  | ~2.94M            | ✗        | 仍可用但已不是 Ultralytics 推薦的首選，YOLO11 在更少參數下精度更高                                                               |
| YOLO11n-seg  | ~2.83M            | ✓ 選這個 | Ultralytics 官方推薦，訓練穩定，文件和社群資源豐富，API 成熟                                                                     |
| YOLOv12n-seg | ~2.86M            | ✗        | 非 Ultralytics 官方產品線，attention-based 架構訓練不穩定、記憶體消耗高、CPU 推論慢。原作者明確建議不要用 Ultralytics 的實作版本 |
| YOLO26n-seg  | 待確認            | ✗ (暫時) | 2026 年 1 月才發布，seg 任務的文件和社群經驗尚不成熟。可作為 ablation 對比實驗                                                   |

原 PRD 寫 YOLOv12 是因為「最新」，但最新不等於最適合。你是第一次做 multi-class instance seg fine-tune，框架的穩定性比多 0.5% 的 mAP 重要一百倍。

---

## Semantic Decoder 設計方向（初版）

你有 U-Net 經驗，所以這個對你是合理的延伸：

```python
class LightweightSemanticDecoder(nn.Module):
    """
    從 YOLO11 backbone 的 FPN P3 feature map 接出
    做 multi-class stuff segmentation
    """
    def __init__(self, in_channels=128, num_stuff_classes=8):
        super().__init__()
        # 輕量化設計：depthwise separable conv 壓參數量
        self.conv1 = DepthwiseSeparableConv(in_channels, 64, kernel_size=3)
        self.conv2 = DepthwiseSeparableConv(64, 32, kernel_size=3)
        self.classifier = nn.Conv2d(32, num_stuff_classes, kernel_size=1)
        # Bilinear upsample 回原圖解析度

    def forward(self, p3_feature):
        x = F.relu(self.conv1(p3_feature))
        x = F.relu(self.conv2(x))
        x = self.classifier(x)
        x = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=False)
        return x  # (B, num_stuff_classes, H, W)
```

這不是最終版，你訓練之後會根據結果調整。重點是：

- 參數量極少（< 1M），不會吃掉 VRAM
- 結構簡單，debug 容易
- 你可以在 ablation study 裡比較不同設計（接 P2 vs P3、加不加 skip connection）

---

## SANPO Class 分組（需要你確認）

根據 SANPO dataset 的語意類別，你需要決定哪些是 Things、哪些是 Stuff：

**Things (Instance Segmentation — YOLO 處理)：**
行人、車輛、自行車、動物、路障、路燈柱、垃圾桶...

**Stuff (Semantic Segmentation — 你的 Decoder 處理)：**
人行道、道路、草地、天空、建築牆面、斑馬線、導盲磚...

具體的 class mapping 要等你深入看 SANPO 的 annotation spec 才能定。
這是 Phase 1 最重要的任務之一。

---

## 12 週暑假時程

### Phase 0：基礎準備（第 1-2 週）

**目標：** 理解工具鏈 + 確定 class mapping

| 任務              | 細節                                                                                                                                                                           | 產出                               |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------- |
| YOLO11 熟悉       | 讀 Ultralytics 文件（segment task），用 COCO val 跑一次推論，觀察輸出格式（bbox, mask coefficients, prototype）                                                                | `notes/yolo11_output_format.md`    |
| SANPO 深入分析    | 搞清楚 SANPO annotation 的完整 class list、RGB encoding 規則。你的 dataset.py 已經解析了 `semantic = R channel`、`instance = G*256 + B`，但你需要知道 semantic id 對應什麼類別 | `configs/sanpo_class_mapping.yaml` |
| Things/Stuff 分組 | 根據 SANPO class list 決定哪些歸 instance seg（Things）、哪些歸 semantic seg（Stuff）                                                                                          | 寫進 config                        |
| 論文精讀          | Kirillov "Panoptic Segmentation" (2019) — 重點看 PQ 定義和 fusion 邏輯                                                                                                         | `notes/literature.md`              |
| 環境確認          | Ultralytics 安裝、確認你的 GPU 能跑 YOLO11n-seg inference                                                                                                                      | 跑通一次推論                       |

---

### Phase 1：資料 Pipeline 完善（第 3-4 週）

**目標：** SANPO → 可訓練的格式

| 任務                        | 細節                                                                                                | 產出                                   |
| --------------------------- | --------------------------------------------------------------------------------------------------- | -------------------------------------- |
| 完成 10-frame decimation    | 你 preprocess script 裡的 `TODO: Pick one frame every 10 frames`                                    | 更新 `SANPO_data_preprocess_script.py` |
| SANPO mask → YOLO polygon   | Instance mask（Things class）轉 YOLO segment txt 格式：用 `cv2.findContours` 從 mask 提取多邊形座標 | `data/scripts/convert_to_yolo.py`      |
| SANPO mask → Semantic label | Stuff class 的 mask 轉成 per-pixel class index 的 numpy array，給你的 semantic decoder 訓練用       | `data/scripts/convert_semantic.py`     |
| Train/Val split             | 按 sequence（影片）分，不按 frame。避免同一段影片的相鄰幀同時出現在 train 和 val（data leakage）    | `data/scripts/split_dataset.py`        |
| Unit tests                  | 測轉換邏輯：polygon 能還原回 mask、class id 正確、split 無重疊                                      | `tests/test_data_pipeline.py`          |
| DVC 設定                    | 大檔案追蹤                                                                                          | `.dvc` 檔案                            |

**這個 Phase 最容易卡的地方：**

- SANPO 的 segmentation mask encoding 可能有你沒預期到的 edge case（比如某些 class 的 instance id 是 0）
- polygon 轉換時，小物件的 contour 可能只有幾個點，YOLO 會不會接受要測試

---

### Phase 2：Instance Seg Baseline + Semantic Decoder（第 5-7 週）

**目標：** 兩個分支都能各自跑起來

#### 第 5 週：Instance Segmentation Baseline

| 任務                  | 細節                                                               |
| --------------------- | ------------------------------------------------------------------ |
| 寫 `sanpo.yaml`       | Ultralytics dataset config，指向轉換後的 YOLO 格式資料             |
| Fine-tune YOLO11n-seg | `model.train(data='sanpo.yaml', epochs=100, imgsz=640)`            |
| 記錄 baseline         | mAP50, mAP50-95 on val set                                         |
| VRAM 監控             | 確認 batch size 和記憶體用量，必要時開 AMP / gradient accumulation |

#### 第 6-7 週：Semantic Decoder 設計與訓練

| 任務                  | 細節                                                                   |
| --------------------- | ---------------------------------------------------------------------- |
| 提取 FPN features     | 寫 hook 從 YOLO11 backbone 的 P3（或 P2）拿 feature map                |
| 實作 Semantic Decoder | 從簡單開始：3 層 conv + bilinear upsample                              |
| 訓練                  | 凍結 backbone，只訓練 decoder。Loss: CrossEntropyLoss on stuff classes |
| 評估                  | Per-class mIoU on stuff classes                                        |
| 迭代                  | 效果不好的話：試接不同 FPN level、加 skip connection、調 channel 數    |

**這裡的 fallback：** 如果 semantic decoder 怎麼調都不行，你可以退回用一個獨立的輕量 semantic seg 模型（如 BiSeNet），雖然不共享 backbone 但至少有結果。

---

### Phase 3：Panoptic Fusion + Depth Risk（第 8-9 週）

**目標：** 把兩個分支的輸出合成完整的 panoptic map，加上距離風險

#### 第 8 週：Panoptic Fusion Module

| 任務             | 細節                                                                          |
| ---------------- | ----------------------------------------------------------------------------- |
| 實作 fusion 邏輯 | Rule: instance mask 優先（Things），剩餘像素填 stuff prediction               |
| 衝突處理         | 多個 instance 重疊 → 取 confidence 高的；instance 與 stuff 重疊 → instance 贏 |
| PQ 評估          | 用你已經寫好的 `panoptic_quality.py` 評估完整 panoptic 結果                   |
| 視覺化           | 畫幾張好的 case 和壞的 case                                                   |

#### 第 9 週：Depth-aware Risk Scoring

| 任務                       | 細節                                           |
| -------------------------- | ---------------------------------------------- |
| 讀取對應 depth map         | 從你的 Dataset class 拿 depth tensor           |
| 算每個 instance 的平均深度 | `mean_depth = depth_map[instance_mask].mean()` |
| 定義風險等級               | 紅 (< 2m), 黃 (2-5m), 綠 (> 5m) — 或連續分數   |
| 距離加權 Recall            | 計算 2m 內障礙物的 recall（PRD 要求 > 95%）    |
| 風險視覺化                 | 輸出帶有顏色標註的推論結果圖                   |

---

### Phase 4：實驗與 Ablation（第 10-11 週）

**目標：** 產出有說服力的數據

| 實驗                | 內容                                           | 要量化的指標     |
| ------------------- | ---------------------------------------------- | ---------------- |
| Exp 1: Model size   | YOLO11n vs YOLO11s — 速度 vs 精度 trade-off    | mAP, FPS, VRAM   |
| Exp 1b: YOLO26 對比 | 若 YOLO26-seg 文件成熟，替換 backbone 比較     | mAP, FPS, PQ     |
| Exp 2: Decoder 設計 | 接 P2 vs P3、有無 skip connection              | Stuff mIoU       |
| Exp 3: Fusion 策略  | Confidence-based vs Area-based                 | PQ, SQ, RQ       |
| Exp 4: 距離風險     | 有無 depth risk scoring 對近距離 recall 的影響 | Recall@2m        |
| Exp 5: Robustness   | 加噪音/模糊/亮度變化後的 PQ 穩定性             | PQ degradation % |
| Failure Analysis    | 挑模型徹底失敗的 case 深度分析原因             | 定性分析         |

**全部結果記錄到 `experiments/` 目錄，用 MLflow 或至少 TensorBoard 追蹤。**

---

### Phase 5：收尾與包裝（第 12 週）

| 任務              | 細節                                            |
| ----------------- | ----------------------------------------------- |
| README.md         | 架構圖、結果表格、安裝指令、使用說明            |
| CI pipeline       | Ruff lint + Pytest（你的 CI template 已經有了） |
| Demo script       | 輸入一張圖 → 輸出 panoptic map + risk overlay   |
| 可選：Gradio demo | 簡單 web UI，上傳圖片看結果                     |
| 程式碼整理        | 移除 debug code，確保結構清晰                   |

---

## 最終目錄結構

```
KODAMA/
├── README.md
├── .github/workflows/ci.yml
├── configs/
│   ├── sanpo.yaml                  # YOLO dataset config
│   └── sanpo_class_mapping.yaml    # Things/Stuff 分組定義
├── data/
│   ├── scripts/
│   │   ├── preprocess.py           # SANPO 前處理 ✅ 已有
│   │   ├── processor.py            # 裁切/resize ✅ 已有
│   │   ├── convert_to_yolo.py      # mask → YOLO polygon
│   │   ├── convert_semantic.py     # mask → per-pixel stuff label
│   │   └── split_dataset.py        # train/val split by sequence
├── models/
│   ├── semantic_decoder.py         # ★ 你自己設計的 lightweight decoder
│   ├── panoptic_fusion.py          # ★ 你自己寫的 fusion 邏輯
│   └── risk_scoring.py             # ★ depth-aware 風險評估
├── metrics/
│   └── panoptic_quality.py         # PQ 評估 ✅ 已有
├── datasets/
│   └── sanpo_dataset.py            # Dataset class ✅ 已有
├── scripts/
│   ├── train_instance.py           # YOLO11 fine-tune
│   ├── train_semantic.py           # Semantic decoder 訓練
│   ├── evaluate.py                 # 完整 panoptic 評估
│   └── inference_demo.py           # 單張推論 + 視覺化
├── tests/
│   ├── test_data_pipeline.py
│   ├── test_semantic_decoder.py
│   ├── test_panoptic_fusion.py
│   └── test_panoptic_quality.py
├── experiments/
│   ├── baseline.md
│   └── ablation.md
├── docs/
│   ├── PRD.md                      # ✅ 已有
│   └── TECH.md                     # ✅ 已有
└── requirements.txt
```

★ = 你的原創模組

---

## 風險與退路

| 風險                          | 觸發條件                           | 退路                                        |
| ----------------------------- | ---------------------------------- | ------------------------------------------- |
| SANPO label 轉換有問題        | mask encoding 有未預期的 edge case | 先處理最乾淨的幾個 sequence，不用全部       |
| Instance seg fine-tune 效果差 | mAP < 30                           | 增加資料增強、多跑 epoch、試 YOLO11s        |
| Semantic decoder 不收斂       | mIoU < 20 after 50 epochs          | 換一個獨立的輕量 seg 模型（BiSeNet）        |
| VRAM 不夠                     | OOM during training                | 降 batch size + gradient accumulation + AMP |
| 時間不夠                      | Phase 4 來不及做完                 | 砍部分 ablation，優先確保 Phase 5 完成      |

---

## Future Work（寫進 README + 面試用）

以下是本專案完成後的自然延伸方向，分短期和長期兩個層次。

### 短期：模型優化與部署（1-2 個月）

**Edge Deployment Pipeline**

YOLO11n-seg（~2.83M 參數）加上 Lightweight Semantic Decoder（< 1M 參數），
整體模型約 4M 參數 / 10 GFLOPs，具備部署至邊緣裝置的條件。

```
PyTorch (.pt)
  → ONNX 匯出（統一中間格式）
    → TensorRT FP16    — NVIDIA Jetson Nano/Xavier，目標 < 20ms/frame
    → CoreML           — iOS，目標 30 FPS on iPhone 12+
    → TFLite + NNAPI   — Android，目標 30 FPS on Pixel 6+
    → NCNN             — Raspberry Pi 5，目標 ~80ms/frame
```

**INT8 量化：** 透過 Post-Training Quantization 進一步壓縮模型至 ~1-2MB，
以極小的精度損失（預估 PQ 下降 < 2%）換取 2-3 倍的推論加速。
需要建立 calibration dataset 並驗證量化後的 PQ / Recall@2m 指標。

**模型升級實驗：** 待 YOLO26-seg 的文件和社群經驗成熟後，
可替換 backbone 進行對比實驗。YOLO26 的 NMS-free 設計可進一步降低
邊緣裝置上的推論延遲。

### 中期：完整視障輔助原型（3-6 個月）

**即時相機串流整合**

```
手機相機 → Frame Buffer → Model Inference → Panoptic Map + Risk
                                               │
                                               ▼
                                     [TTS Engine] → 語音警告
                                     「前方兩公尺有行人」
                                     「右側一公尺有路障」
```

- 相機前處理：即時 crop + resize + normalize，需處理不同手機的解析度和 aspect ratio
- 推論排程：frame skipping 策略（不需要每幀都跑），在精度和電量間取平衡
- 結果後處理：temporal smoothing，避免風險等級在相鄰幀之間跳動

**語音回饋系統**

- 將 risk scoring 結果轉為自然語言描述
- 優先級佇列：近距離高風險物體優先播報
- 空間音效：利用立體聲暗示障礙物方位（左/中/右）

**觸覺回饋（震動）**

- 手機震動模式編碼距離和方向
- 低延遲，適合不方便戴耳機的場景

### 長期：研究方向延伸

**Temporal Panoptic Segmentation**

- 利用影片的時序資訊做跨幀 instance tracking + 速度估計
- 從「這裡有障礙物」升級到「這個障礙物正以 X m/s 靠近」
- 可參考 Video Panoptic Segmentation (VPS) 相關工作

**自監督 Depth Estimation**

- 移除對 ground-truth depth map 的依賴
- 用 monocular depth estimation（如 Depth Anything V2）取代 SANPO 提供的 depth
- 使模型能泛化到任意場景，不限於 SANPO 的資料分布

**多感測器融合**

- 結合 IMU（慣性測量單元）資料估計使用者的行走方向和速度
- 結合 GPS 做路徑規劃和導航
- 長期目標：從「感知」走向「導航」

---

## 每週自我檢查

1. 這週有沒有新的程式碼跑起來（不是寫了但沒測）？
2. 有 commit + push 到 GitHub 嗎？
3. 遇到的問題有記錄嗎？
4. 下週最重要的一件事是什麼？
5. 有沒有陷入「一直在讀但沒在做」的狀態？
