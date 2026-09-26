import random

from laya_omni.data import load_teacher, shuffled, teacher_key


def test_shuffle_keeps_teacher_probs_with_their_options():
    q = {"type": "choice", "instructions": "Which?", "criteria": ["cat", "dog", "bird", "fish"]}
    probs = [0.1, 0.6, 0.2, 0.1]
    for seed in range(20):
        q2, y, p2 = shuffled(q, 1, random.Random(seed), probs)
        assert q2["criteria"][y] == "dog"
        assert dict(zip(q2["criteria"], p2)) == dict(zip(q["criteria"], probs))


def test_noul_is_not_shuffled():
    q = {"type": "noul", "instructions": "Is it?"}
    assert shuffled(q, 1, random.Random(0), [0.3, 0.7]) == (q, 1, [0.3, 0.7])
    assert shuffled(q, 1, random.Random(0)) == (q, 1)


def test_teacher_lookup(tmp_path):
    row = {"image": "a/1.jpg", "question": {"type": "choice", "instructions": "Which?", "criteria": ["x", "y"]}, "label": 0}
    path = tmp_path / "t.jsonl"
    path.write_text('{"image": "a/1.jpg", "question": {"type": "choice", "instructions": "Which?", "criteria": ["x", "y"]}, '
                    '"label": 0, "teacher": {"probs": [0.8, 0.2]}}\n')
    assert load_teacher([path])[teacher_key(row)] == [0.8, 0.2]
