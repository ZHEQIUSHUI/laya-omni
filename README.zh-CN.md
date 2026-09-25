# laya-omni

**给 Laya 加上眼睛和耳朵：图像、音频可选输入的类型化决策模型。**

[Laya](https://github.com/NandhaKishorM/laya) 对文本状态回答受约束的问题（`choice` 选择、
`score` 分级、`noul` 是非），一次前向完成，给出校准过的概率，不生成任何 token。
laya-omni 让状态里还可以带上图片、音频，或者两者都有；除此之外 Laya 的一切保持不变：
同样的问题格式、同样的输出，不给图像和音频时，结果与原版 Laya 逐位一致。

[English](README.md) · [架构](docs/architecture.md) · [完整结果](docs/results.md) · [复现](docs/training.md)

## 能做什么

`formal-v1` 在从未训练过的测试题上的准确率，对照的是同样的题目交给不看图、不听音的原版 Laya：

| | 原版 Laya（只看文字） | laya-omni |
|---|---:|---:|
| 照片问答（VQAv2 / GQA） | 0.45 / 0.55 | **0.77 / 0.77** |
| 读图中文字（TextVQA / OCR-VQA） | 0.36 / 0.44 | **0.81 / 0.90** |
| 图表与文档（ChartQA / DocVQA） | 0.36 / 0.40 | **0.80 / 0.71** |
| 计数（TallyQA） | 0.26 | **0.88** |
| 声音事件（ESC-50 / VGGSound） | 0.43 / 0.43 | **0.96 / 0.84** |
| 语音意图（SLURP 101 种 / MINDS-14 含中文等 14 种语言） | 0.34 / 0.41 | **0.95 / 0.96** |
| 音频描述匹配（AudioCaps） | 0.46 | **0.91** |
| 图像 + 音频同时输入（OmniInstruct） | 0.46 | **0.84** |
| 零样本：Mini-ImageNet / MMAU / Song Describer | 0.52 / 0.29 / 0.46 | **0.80 / 0.44 / 0.70** |

完整结果、温度校准、少样本适配，以及试过但行不通的方案：[docs/results.md](docs/results.md)。

## 体量与速度

| | 参数量 |
|---|---:|
| Laya multilingual（冻结） | 约 3.1 亿 |
| SigLIP 2 base/16 图像编码器（冻结） | 约 0.93 亿 |
| Qwen3-ASR-0.6B 音频编码器（冻结） | 1.86 亿 |
| laya-omni 融合模块（训练） | 约 700 万 |

在 RTX 4090 上，带一张图或一段 10 秒音频回答一个问题约 30–120 ms（含编码器）。
图片默认 64 个 token；`detail=True` 用 256 个，读图中文字时更准。

## 使用

```python
from laya_omni import Omni

omni = Omni.load(
    "runs/formal-v1-stage2", laya="models/laya-multilingual",
    image_encoder="models/siglip2-base-patch16-256",
    audio_encoder="models/qwen3-asr-0.6b-audio-encoder",
)

omni.predict("Image.", {
    "sport": {"type": "choice", "instructions": "他们在玩什么运动？",
              "criteria": ["棒球", "网球", "足球"]},
    "crowd": {"type": "noul", "instructions": "有观众吗？"},
}, image="photo.jpg")

omni.predict("Audio clip.", {
    "intent": {"type": "choice", "instructions": "来电者想办什么业务？",
               "criteria": ["查询余额", "冻结银行卡", "缴费"]},
}, audio="call.wav")

omni.predict(state, questions)   # 不给图像和音频：就是原版 Laya
```

多张图片：`image=[a, b]`。输出与 Laya 相同（`choice`、`probabilities`、`score`、`noul`、`confidence`、`action`）。

## 网页演示

```bash
pip install -e '.[web]'
laya-omni serve --fusion runs/formal-v1-stage2 --laya models/laya-multilingual \
    --image-encoder models/siglip2-base-patch16-256 \
    --audio-encoder models/qwen3-asr-0.6b-audio-encoder --examples data --port 8030
```

可以上传图片或音频（也可以用麦克风录音）、自己写问题，并对比「看图听音后」和「不看不听」的概率。

## 原理

冻结的编码器把每张图变成 64（或 256）个特征帧，每秒音频变成约 13 个；一个小 MLP
把每一帧映射成 Laya 输入里的一个「词」，放在 `[CLS]` 之后。Laya 冻结的编码器和决策头
像读文字一样把它们和问题、选项一起读，只在带图像或音频的样本上叠加 LoRA 增量。
每个设计决定的来由见 [docs/architecture.md](docs/architecture.md)。

## 状态

早期研究版本，权重尚未发布。已知短板：科学示意图（AI2D、TQA）、空间关系（VSR），
以及需要推演「接下来会怎样」的任务（如游戏走法），它擅长的是判断「画面 / 声音里有什么」。

## 许可与致谢

Apache-2.0，见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。基于 [Laya](https://github.com/NandhaKishorM/laya)（Convai Innovations）、
[SigLIP 2](https://huggingface.co/google/siglip2-base-patch16-256) 与 [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)，
训练数据为 [docs/training.md](docs/training.md) 所列的公开数据集（各自遵循其许可证），本仓库不分发数据集。
