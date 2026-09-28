# laya-omni

**给 Laya 加上眼睛和耳朵：图像、音频可选输入的类型化决策模型。**

[Laya](https://github.com/NandhaKishorM/laya) 对文本状态回答受约束的问题（`choice` 选择、
`score` 分级、`noul` 是非），一次前向完成，给出校准过的概率，不生成任何 token。
laya-omni 让状态里还可以带上图片、音频，或者两者都有；除此之外 Laya 的一切保持不变：
同样的问题格式、同样的输出，不给图像和音频时，结果与原版 Laya 逐位一致。

[English](README.md) · [架构](docs/architecture.md) · [完整结果](docs/results.md) · [场景微调](docs/finetune.zh-CN.md) · [复现](docs/training.md)

## 能做什么

在标准测试集上和小尺寸 VLM 对比：同一份题、同一张 RTX 4090、一次问一题，计时从读入图片或音频文件到给出答案。每格为「准确率 / 每题耗时中位数」：

| 测试集（随机水平） | laya-omni | Qwen3.5-0.8B | Qwen3.5-2B | Valen（Qwen3.5-2B） |
|---|---:|---:|---:|---:|
| MMBench 英文 dev（0.25） | 0.63 / **21 ms** | 0.75 / 42 ms | **0.84** / 46 ms | 0.73 / 70 ms |
| MMBench 中文 dev（0.25） | 0.58 / **21 ms** | 0.75 / 42 ms | **0.82** / 45 ms | 0.73 / 70 ms |
| MMStar（0.25） | 0.36 / **21 ms** | 0.46 / 42 ms | **0.53** / 46 ms | 0.44 / 70 ms |
| MME，是非题（0.5） | 0.66 / **23 ms** | 0.76 / 48 ms | **0.82** / 62 ms | 0.61 / 105 ms |
| POPE，物体幻觉（0.5） | 0.83 / **22 ms** | 0.88 / 44 ms | **0.90** / 51 ms | 待测 |
| SEED-Bench 图像（0.25） | 0.54 / **26 ms** | **0.72** / 61 ms | 待测 | 待测 |
| Valen-Eval-General-5k（Valen 自己的评测集） | 0.61 / **26 ms** | 0.70 / 50 ms | **0.74** / 61 ms | **0.74** / 207 ms* |
| OmniBench，图像 + 音频（0.25） | 0.40 | 待测（Qwen2.5-Omni-3B） | – | – |
| MMAU test-mini，音频（约 0.25） | 0.45 | 待测（Qwen2.5-Omni-3B） | – | – |

laya-omni 的速度约为最小的 Qwen3.5 的两倍，并且支持音频；但在这些测试集上准确率低 5–18 个点。缩小这个差距（用更大的 VLM 做蒸馏）是目前正在做的事。说明：

- Qwen3.5 的答题方式：一次前向后取选项字母（或 Yes / No）的 logit，不做生成解码，这是它最快的用法。
- Valen 用的是它唯一公开的检查点 [Valen-Preview-0923](https://huggingface.co/Valen-Team/Valen-Preview-0923)（在推箱子任务上继续训练过），在它锁定版本的原版环境里运行；我们的评测脚本复现了它官方评测的结果（0.738 对 0.737）。*这一格是它装上线性注意力加速内核之前测的。Valen 评测集里约 12% 的题目也出现在 laya-omni 的训练数据中。
- 耗时是和其他任务共用机器时测的中位数，干净的单独测速还没做。
- 复现：`scripts/convert_bench.py` 和 `scripts/bench.py`，细节见 [docs/results.md](docs/results.md#standard-benchmarks)。

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
    "runs/formal-v2", laya="models/laya-multilingual",
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
laya-omni serve --fusion runs/formal-v2 --laya models/laya-multilingual \
    --image-encoder models/siglip2-base-patch16-256 \
    --audio-encoder models/qwen3-asr-0.6b-audio-encoder --examples data --port 8030
```

可以上传图片或音频（也可以用麦克风录音）、自己写问题，并对比「看图听音后」和「不看不听」的概率。

## 迁移到你的场景

每个答案放一个文件夹，一条命令即可：从通用模型出发训练，自动早停和校准，几百个样本几分钟完成。

```bash
python scripts/finetune.py --folders my_task/ --question "这个零件有缺陷吗？" \
    --base runs/formal-v2 --laya models/laya-multilingual \
    --image-encoder models/siglip2-base-patch16-256 --out runs/my-task
```

教程与实测结果：[docs/finetune.zh-CN.md](docs/finetune.zh-CN.md)。

## 原理

冻结的编码器把每张图变成 64（或 256）个特征帧，每秒音频变成约 13 个；一个小 MLP
把每一帧映射成 Laya 输入里的一个「词」，放在 `[CLS]` 之后。Laya 冻结的编码器和决策头
像读文字一样把它们和问题、选项一起读，只在带图像或音频的样本上叠加 LoRA 增量。
每个设计决定的来由见 [docs/architecture.md](docs/architecture.md)。

## 状态

早期研究版本。权重：[zheqiushui/laya-omni](https://huggingface.co/zheqiushui/laya-omni)（融合模块 + 音频编码器）。已知短板：科学示意图（AI2D、TQA）、空间关系（VSR），
以及需要推演「接下来会怎样」的任务（如游戏走法），它擅长的是判断「画面 / 声音里有什么」。

## 许可与致谢

代码采用 Apache-2.0，见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。已发布的权重由公开数据集训练而成（数据集本身不在本仓库分发），其中部分数据集仅限非商业研究使用（如 ScienceQA、ESC-50、Hateful Memes），因此权重请用于研究与非商业用途；如需商用，请用本仓库的训练配方在许可允许商用的数据上重新训练。数据集清单与许可说明见 [docs/training.md](docs/training.md)。

基于 [Laya](https://github.com/NandhaKishorM/laya)（Convai Innovations）、[SigLIP 2](https://huggingface.co/google/siglip2-base-patch16-256) 与 [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)，均为 Apache-2.0。
