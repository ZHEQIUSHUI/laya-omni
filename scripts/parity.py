"""Check laya-omni's text path against the official `laya` package on the same checkpoint.

Compares token sequences, fp32 logits and act logits of the two models on one
batch, then the published answers of both runtimes (official uses its default
autocast; ours runs fp32).

    python scripts/parity.py models/laya-multilingual
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

import torch

from laya_omni.agent import Agent, collate_items

CASES = [
    (
        "Invoice 4411 was charged twice. Please refund the duplicate today or we will cancel.",
        {
            "department": {"type": "choice", "instructions": "Which team should handle this?",
                           "criteria": {"billing": "charges and refunds", "technical": "bugs", "sales": "pricing", "other": "anything else"}},
            "urgency": {"type": "score", "instructions": "How urgent is it?", "criteria": ["routine", "soon", "today"]},
            "refund": {"type": "noul", "instructions": "Does the user explicitly ask for a refund?"},
        },
    ),
    (
        "发票4411重复扣款，请今天退还多扣的金额，否则我们会取消服务。",
        {
            "department": {"type": "choice", "instructions": "这条请求应由哪个团队处理？",
                           "criteria": {"billing": "扣款与退款", "technical": "故障报错", "sales": "价格采购"}},
            "urgency": {"type": "score", "instructions": "这条请求有多紧急？", "criteria": ["常规", "尽快", "今天必须解决"]},
            "refund": {"type": "noul", "instructions": "用户是否明确要求退款？"},
        },
    ),
    (
        "Stay alive and eat.",
        {"move": {"type": "choice", "instructions": "Choose the best safe action.",
                  "criteria": {"up": "Wrong way.", "down": "Safe. Best move.", "left": "Blocked. Wall.", "east": "Unsafe. Traps the snake."}}},
    ),
    (
        {"user": "delete all production tables now", "role": "intern", "approved": False},
        {"allow": {"type": "noul", "instructions": "Should this action be allowed to run?"},
         "risk": {"type": "score", "instructions": "Risk level", "criteria": ["low", "medium", "high", "critical"]},
         "one": {"type": "choice", "instructions": "Only option", "criteria": ["refuse"]}},
    ),
]


def main():
    import laya

    src = Path(sys.argv[1])
    ours = Agent(src, dtype="float32")
    with tempfile.TemporaryDirectory() as tmp:
        # Official loading rewrites tokenizer_config.json in place; give it a copy.
        official_dir = Path(tmp) / "ckpt"
        shutil.copytree(src, official_dir, ignore=shutil.ignore_patterns(".cache"))
        official = laya.load(str(official_dir), device=str(ours.device))

        worst = {"logits": 0.0, "act": 0.0, "prob": 0.0}
        same_choice = total = 0
        for state, questions in CASES:
            items, _ = ours.prepare(state, questions)
            theirs_items = []
            for qid in questions:
                q = official._to_internal(questions[qid])
                ids, markers = laya.common.build_sequence(official.tok, state, q, official.cfg.get("max_len", 512), official.cfg.get("head_max_len", 192))
                theirs_items.append((ids, markers))
            assert [(i["ids"], i["markers"]) for i in items] == theirs_items, "token sequences differ"

            b = {k: v.to(ours.device) for k, v in collate_items(items, ours.tok.pad_token_id).items()}
            with torch.no_grad():
                a_logits, a_act = ours.model(**b)
                official.model.float()
                o_logits, o_act = official.model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])
            mask = b["marker_mask"]
            worst["logits"] = max(worst["logits"], float((a_logits - o_logits)[mask].abs().max()))
            worst["act"] = max(worst["act"], float((a_act - o_act).abs().max()))

            mine = ours.predict(state, questions)["answers"]
            official.model.to(official.device)
            theirs = official.predict(state, questions)["answers"]
            for qid in questions:
                total += 1
                pa, po = mine[qid].get("probabilities"), theirs[qid].get("probabilities")
                if pa:
                    worst["prob"] = max(worst["prob"], max(abs(pa[k] - po[k]) for k in pa))
                    same_choice += max(pa, key=pa.get) == max(po, key=po.get)
                else:
                    worst["prob"] = max(worst["prob"], abs(mine[qid]["noul"] - theirs[qid]["noul"]))
                    same_choice += (mine[qid]["noul"] > 0.5) == (theirs[qid]["noul"] > 0.5)
            print(json.dumps({"state": str(state)[:40], "ours": mine, "official": theirs}, ensure_ascii=False)[:600])
        print(f"\nfp32 max |logit diff| {worst['logits']:.2e}, max |act diff| {worst['act']:.2e}")
        print(f"published answers: same decision {same_choice}/{total}, max prob diff {worst['prob']:.4f} (official uses autocast)")


if __name__ == "__main__":
    main()
