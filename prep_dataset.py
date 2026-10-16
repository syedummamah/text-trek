import json, csv, hashlib, os, re, sys
SRC = sys.argv[1] if len(sys.argv) > 1 else "Multilingual_E-Sahulat_AI_Hackathon_Dataset.txt"
os.makedirs("data", exist_ok=True)
rows = [json.loads(l) for l in open(SRC, encoding="utf-8") if l.strip().startswith("{")]

# 1) seed passages = unique gold evidence (EN + UR), one passage per (url, evidence)
seen, passages, ev2pid = {}, [], {}
for r in rows:
    if not r["answerable"] or not r.get("gold_evidence_en"):
        continue
    key = (r["source_url"], r["gold_evidence_en"])
    if key not in seen:
        pid = "G%03d" % (len(seen) + 1)
        seen[key] = pid
        doc = r["source_url"].rstrip("/").split("/")[-1].replace(".php", "")
        passages.append({"doc_id": doc, "url": r["source_url"], "passage_id": pid,
                         "text": r["gold_evidence_en"] + " || " + (r.get("gold_evidence_ur") or ""),
                         "date_accessed": "2026-10-16", "origin": "gold_seed"})
    ev2pid[r["qid"]] = seen[key]
json.dump(passages, open("data/passages_seed.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# 2) test_queries.csv in our eval format
lang = {"en": "english", "ur": "urdu", "roman_ur": "roman"}
with open("data/test_queries.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["query_id", "query", "language", "topic", "answerable", "expected_passage_id",
                "split", "group_id", "expected_behavior", "expected_answer_en", "expected_answer_ur", "source_url"])
    for r in rows:
        w.writerow([r["qid"], r["query"], lang.get(r["language"], r["language"]), r["topic"],
                    int(r["answerable"]), ev2pid.get(r["qid"], ""),
                    "dev" if r["split"] == "train_dev" else "test", r["intent_id"],
                    r.get("expected_behavior") or "", r.get("expected_answer_en") or "",
                    r.get("expected_answer_ur") or "", r.get("source_url") or ""])
print(len(rows), "queries,", len(passages), "seed passages")