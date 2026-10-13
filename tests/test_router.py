import json

import pytest

import router


def labelled(qid, split, label, query, correct, usd=0.001, rewrite_usd=0.0, all_failed=False, type_="single"):
    return {"id": qid, "type": type_, "split": split, "group": qid, "query": query, "rewrite_usd": rewrite_usd,
            "rewrite_s": 0.0, "label": label, "all_failed": all_failed,
            "routes": {r: {"correct": correct[r], "usd": usd * (i + 1), "local_ms": 10.0 * i, "llm_s": 1.0, "chunks": []}
                       for i, r in enumerate(router.ROUTES)}}


def test_router_learns_and_falls_back_to_deep_when_unsure():
    texts = ["what is the capital of france"] * 5 + ["how many vacation days do employees get"] * 5 + \
            ["compare the edge server cost and the robot limit"] * 5
    labels = ["none"] * 5 + ["light"] * 5 + ["deep"] * 5
    confident = router.Router(threshold=0.0).fit(texts, labels)
    assert confident.route(["what is the capital of spain", "how many sick days do employees get"]) == ["none", "light"]
    unsure = router.Router(threshold=0.99).fit(texts, labels)
    assert unsure.route(["what is the capital of spain"]) == ["deep"]


def test_score_charges_rewrites_only_where_they_are_used():
    rows = [labelled("a", "test", "none", "q", {"none": True, "light": True, "deep": True}, rewrite_usd=1.0),
            labelled("b", "test", "deep", "q", {"none": False, "light": False, "deep": True}, rewrite_usd=1.0)]
    never = router.score(rows, ["none", "none"], charge_rewrite=False)
    routed = router.score(rows, ["none", "deep"], charge_rewrite=True)
    assert never["accuracy"] == 0.5 and routed["accuracy"] == 1.0
    assert never["usd_per_1k"] == pytest.approx(1000 * (0.001 + 0.001) / 2)          # no rewrite without retrieval
    assert routed["usd_per_1k"] == pytest.approx(1000 * (0.001 + 1.0 + 0.003 + 1.0) / 2)  # router always rewrites


def test_group_bootstrap_moves_whole_groups():
    rows = [{"group": g} for g in "aabbcc"]
    mean, ci = router.group_bootstrap(rows, [1, 0, 1, 0, 1, 0])
    assert mean == 0.5 and ci == [0.5, 0.5]  # every group is 1/2, so resampling groups can't move the mean


def test_evaluate_reports_baselines_oracle_and_router_accuracy(tmp_path, monkeypatch):
    rows = []
    for i in range(12):
        split = "train" if i < 9 else "test"
        rows.append(labelled(f"g{i}", split, "none", f"what is the capital of country {i}",
                             {"none": True, "light": False, "deep": False}, type_="general"))
        rows.append(labelled(f"c{i}", split, "light", f"how many vacation days policy {i}",
                             {"none": False, "light": True, "deep": True}))
        rows.append(labelled(f"x{i}", split, "deep", f"which documents took effect on date {i}",
                             {"none": False, "light": False, "deep": False}, all_failed=True))
    monkeypatch.setattr(router, "RESULTS", tmp_path)
    monkeypatch.setattr(router, "questions", lambda: [{"id": r["id"]} for r in rows])
    (tmp_path / "labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    report = router.evaluate(threshold=0.0)
    systems = report["systems"]
    assert systems["oracle"]["accuracy"] == systems["router"]["accuracy"] == pytest.approx(2 / 3, abs=1e-3)
    assert systems["never retrieve"]["accuracy"] == pytest.approx(1 / 3, abs=1e-3)
    assert systems["oracle"]["routes"]["none"] == 6  # every-route-fails questions go to the free route
    assert report["router_accuracy"]["all_failed_left_out"] == 3 and report["router_accuracy"]["overall"] == 1.0
    assert report["company_questions_only"]["systems"]["always light"]["n"] == 6
    assert (tmp_path / "routing.json").exists() and (tmp_path / "misroutes.csv").exists()


def test_evaluate_refuses_labels_with_missing_questions(tmp_path, monkeypatch):
    rows = [labelled("a", "train", "none", "q", {"none": True, "light": True, "deep": True}),
            labelled("b", "test", "none", "q", {"none": True, "light": True, "deep": True})]
    monkeypatch.setattr(router, "RESULTS", tmp_path)
    monkeypatch.setattr(router, "questions", lambda: [{"id": "a"}, {"id": "b"}, {"id": "c"}])
    (tmp_path / "labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(SystemExit, match="1 questions have no row"):
        router.evaluate(threshold=0.0)


def test_spotcheck_never_overwrites_hand_marks(tmp_path, monkeypatch):
    monkeypatch.setattr(router, "RESULTS", tmp_path)
    (tmp_path / "spotcheck.csv").write_text("id,label_ok\np3-001,1\n")
    with pytest.raises(SystemExit, match="already has hand marks"):
        router.spotcheck(5, 0)


def test_oracle_uses_measured_cost_instead_of_route_depth():
    row = labelled("a", "test", "none", "q", {"none": True, "light": True, "deep": True})
    row["routes"]["none"]["usd"] = 0.02
    row["routes"]["deep"]["usd"] = 0.0001
    result = router.compare([row], ["none"])["systems"]["oracle"]
    assert result["routes"] == {"deep": 1}
    assert result["usd_per_1k"] == 0.1
    for route in router.ROUTES:
        row["routes"][route]["correct"] = False
    row["all_failed"] = True
    assert router.compare([row], ["none"])["systems"]["oracle"]["routes"] == {"deep": 1}


def test_evaluate_rejects_duplicate_question_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(router, "RESULTS", tmp_path)
    row = labelled("a", "train", "none", "q", {"none": True, "light": True, "deep": True})
    (tmp_path / "labels.jsonl").write_text((json.dumps(row)+"\n")*2)
    with pytest.raises(SystemExit, match="duplicate question IDs"):
        router.evaluate(0.5)
