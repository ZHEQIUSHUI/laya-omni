# 场景微调教程

[English](finetune.md)

laya-omni 的通用模型能回答「画面 / 声音里有什么」这类问题，但每个实际场景都有自己的判断标准：
产线上的零件算不算缺陷、监控画面里的行为是否异常、客服录音属于哪类工单……这些用几百到几千个
标注样本微调一下，通常就能达到可用的精度。

微调只训练约 700 万参数的融合模块，底座（Laya、SigLIP 2、Qwen3-ASR）不动。每个场景得到一个
约 28MB 的目录，多个场景共用同一套底座。

## 1. 准备数据

最简单的方式：**每个答案一个文件夹**。

```
my_task/
├── 正常/        ← 文件夹名就是选项
│   ├── 0001.jpg
│   └── ...
└── 缺陷/
    ├── 0101.jpg
    └── ...
```

- 图片（jpg、png、webp、bmp）和音频（wav、flac、mp3、ogg、m4a）都可以，一个任务里用一种即可。
- 文件夹名就是选项。只有两个文件夹，且名字是 `是 / 否`、`yes / no` 或 `true / false` 时，会按是非题（noul）来问。
- **数量建议**：每个选项至少 100 个，300–1000 个更稳。选项之间数量尽量接近。

也可以用 CSV（`file,label` 两列，可选 `question`、`state` 列，每一行可以问不同的问题），
或者直接写 laya-omni 的 jsonl（见文末）。

## 2. 一条命令微调

```bash
python scripts/finetune.py \
    --folders my_task/ --question "这个零件有缺陷吗？" \
    --base runs/formal-v2 --laya models/laya-multilingual \
    --image-encoder models/siglip2-base-patch16-256 \
    --audio-encoder models/qwen3-asr-0.6b-audio-encoder \
    --out runs/my-task
```

它会依次：

1. 按选项分层，随机留出 20% 作验证集（`--val-fraction`）；
2. 音频任务先缓存音频特征；
3. 从通用模型 `--base` 出发训练（默认 8 个 epoch），**每个 epoch 都在验证集上评估，保留 NLL 最低的那一轮**（早停）；
4. 在验证集上拟合温度（校准），写入 `runs/my-task/fusion_config.json`；
5. 打印验证集的准确率，以及不看图 / 不听音时的准确率作为对照。

单张 4090 上，几百个样本通常只需几分钟。

### 保住通用能力：回放

微调后如果还想让同一个模型回答通用问题，训练时混入一些通用数据：

```bash
    --replay 1000 \
    --replay-jsonl data/cauldron/vqav2.jsonl --replay-jsonl data/audio/esc50.jsonl \
    --audio-features cache/qwen3-asr
```

每个 epoch 混入约 1000 道通用题。只在一个场景里使用的模型可以不加。

## 3. 使用

```python
from laya_omni import Omni

omni = Omni.load("runs/my-task", laya="models/laya-multilingual",
                 image_encoder="models/siglip2-base-patch16-256")
omni.predict("Image.", {"defect": {"type": "noul", "instructions": "这个零件有缺陷吗？"}},
             image="new_part.jpg")
```

问法和训练时保持一致效果最好；`predict` 仍然可以问其他问题（加了回放时效果更好）。

## 4. 效果参考

在通用训练里从没出现过的贪吃蛇游戏画面上（4 选 1，随机猜 0.25），同样设置下的最佳测试准确率：

| 训练样本 | 从通用模型出发（formal-v1） | 从零开始 |
|---:|---:|---:|
| 100 | 0.36 | 0.38 |
| 500 | **0.55** | 0.35 |
| 2000 | **0.68** | 0.26 |

用 `finetune.py` 实测：

- 贪吃蛇 500 帧（400 训练 / 100 验证）+ 回放 VQAv2 和 ESC-50：验证 0.55，ESC-50 仍是 0.96（微调前 0.955），通用能力没掉；
- 咳嗽 vs 笑声 200 段音频（160 训练 / 40 验证）：验证 1.00。

「画面 / 声音里有什么」这类任务学得很快；贪吃蛇这类要按规则推演的任务需要更多样本。

## 5. 常见问题

- **验证集准确率不涨**：先看不看图时的准确率（对照）。两者接近，说明答案可能不在画面里，或者需要推演（比如「接下来会发生什么」），这类任务需要更多样本。
- **验证准确率不错，NLL 却很高**：小样本容易过度自信。早停已经选了 NLL 最好的一轮，校准也会再纠正一次；样本再多一些效果更好。
- **图里有小字、表格、界面**：推理时加上 `detail=True`（256 个 token）。
- **一个场景多个问题**：用 CSV 的 `question` 列或 jsonl，每行问自己的问题，一个适配器就能覆盖一个场景里的几种判断。

## 附：jsonl 格式

每行一道题：

```json
{"state": "Image.", "question": {"type": "choice", "instructions": "这是哪种零件？", "criteria": ["齿轮", "轴承", "螺母"]}, "label": 1, "image": "imgs/0001.jpg"}
```

- `label` 是正确选项的下标（noul 题：1 = 是，0 = 否）；
- 音频用 `"audio": "clips/0001.wav"`；多张图用 `"images": ["a.jpg", "b.jpg"]`；
- 可以加 `"split": "train"` 或 `"test"` 自己指定验证集；
- `state` 里可以放文字背景，比如工单内容、传感器读数，模型会把它和图片、音频一起考虑。
