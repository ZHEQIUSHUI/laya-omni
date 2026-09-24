"""Build the pixels-only game dataset: frames labelled by the deterministic planners.

The question text names the options and nothing else, so the answer is only in
the frame. Rollouts mix in random actions (epsilon) so the states cover more than
the planner's own comfortable path; the label is always the planner's move for
the state shown.

    python scripts/make_game_data.py --out data/games --episodes 400
"""

import argparse
import json
import random
from pathlib import Path

from laya_omni.games import render
from laya_omni.games.bird import BirdGame
from laya_omni.games.bricks import ACTIONS as BRICK_ACTIONS
from laya_omni.games.bricks import BricksGame
from laya_omni.games.snake import DIRECTIONS, SnakeGame

# Several phrasings and label sets per game; training draws one per sample and shuffles
# the option order, so the adapter has to read the question instead of mapping a
# frame to an option slot. Labels are listed in canonical action order.
QUESTIONS = {
    "snake": {
        "instructions": [
            "Look at the screen. Which way should the snake move to stay alive and reach the food?",
            "Where should the snake go next?",
            "Pick the snake's next move: stay safe and head for the food.",
            "Which direction is the best next step for the snake?",
            "蛇下一步应该往哪走？",
        ],
        "labels": [
            ["up", "down", "left", "right"],
            ["north", "south", "west", "east"],
            ["move up", "move down", "move left", "move right"],
            ["上", "下", "左", "右"],
        ],
    },
    "bird": {
        "instructions": [
            "Look at the screen. Should the bird flap up or glide down to fly through the next gap?",
            "What should the bird do now to pass the next opening?",
            "Choose the bird's action for this moment.",
            "小鸟现在应该怎么做才能穿过下一个缺口？",
        ],
        "labels": [
            ["up", "down"],
            ["flap", "glide"],
            ["rise", "sink"],
            ["往上", "往下"],
        ],
    },
    "bricks": {
        "instructions": [
            "Look at the screen. How should the paddle move to catch the ball?",
            "Where should the paddle go so the ball does not fall?",
            "Choose the paddle's next move.",
            "挡板应该怎么移动才能接住球？",
        ],
        "labels": [
            ["left", "right", "hold"],
            ["move left", "move right", "stay"],
            ["west", "east", "wait"],
            ["左移", "右移", "不动"],
        ],
    },
}
STATE = "Game screen."


def snake_episode(seed, eps, max_steps):
    rng = random.Random(seed)
    game = SnakeGame(seed=seed, initial_length=rng.randint(3, 8))
    while game.alive and not game.won and game.ticks < max_steps:
        best = game.preferred()
        if best is None:
            return
        yield render.snake(game), DIRECTIONS.index(best)
        legal = [m.direction for m in game.moves() if m.safe] or [best]
        game.step(rng.choice(legal) if rng.random() < eps else best)


def bird_episode(seed, eps, max_steps):
    rng = random.Random(seed)
    game, history = BirdGame(seed=seed), []
    while game.alive and game.steps < max_steps:
        plan = game.plan()
        yield render.bird(game, history), ["up", "down"].index(plan["autopilot"])
        flap = plan["autopilot"] == "up"
        if rng.random() < eps and plan["safe_flap"] and plan["safe_hold"]:
            flap = not flap
        history.append(game.y)
        game.step(flap)


def bricks_episode(seed, eps, max_steps):
    rng = random.Random(seed)
    game, history = BricksGame(seed=seed), []
    game.paddle_x = rng.uniform(2, 18)
    while game.alive and not game.won and game.steps < max_steps:
        best = game.plan()["best"]
        yield render.bricks(game, history), BRICK_ACTIONS.index(best)
        history.append((game.ball_x, game.ball_y))
        game.step(rng.choice(BRICK_ACTIONS) if rng.random() < eps else best)


EPISODES = {"snake": snake_episode, "bird": bird_episode, "bricks": bricks_episode}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/games")
    ap.add_argument("--games", default="snake,bird,bricks")
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--eps", type=float, default=0.15)
    ap.add_argument("--keep", type=float, default=0.35, help="fraction of frames kept (consecutive frames are near-duplicates)")
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--questions-only", action="store_true", help="only rewrite questions.json")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "questions.json").write_text(json.dumps({"state": STATE, "questions": QUESTIONS}, indent=2, ensure_ascii=False) + "\n")
    if args.questions_only:
        return
    for name in args.games.split(","):
        (out / name).mkdir(parents=True, exist_ok=True)
        rows, counts = [], [0] * len(QUESTIONS[name]["labels"][0])
        for ep in range(args.episodes):
            seed = args.seed_offset + ep
            pick = random.Random(seed * 7919 + 1)
            for t, (img, label) in enumerate(EPISODES[name](seed, args.eps, args.max_steps)):
                if pick.random() > args.keep:
                    continue
                path = f"{name}/{seed:05d}_{t:04d}.png"
                img.save(out / path)
                rows.append({"game": name, "image": path, "seed": seed, "label": label})
                counts[label] += 1
        # Episodes split by seed so near-duplicate frames never straddle train and test.
        with open(out / f"{name}.jsonl", "w") as f:
            for r in rows:
                r["split"] = "test" if r["seed"] % 10 == 0 else "train"
                f.write(json.dumps(r) + "\n")
        print(name, len(rows), "frames; label counts", dict(zip(QUESTIONS[name]["labels"][0], counts)))


if __name__ == "__main__":
    main()
