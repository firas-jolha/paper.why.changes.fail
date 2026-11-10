# Load ReDef (Nam et al., FSE 2026) from its released files: data/redef/1_train.jsonl, 1_valid.jsonl, 1_test.jsonl
import json
rows=[]
for split in ["train","valid","test"]:
    with open(f"data/redef/1_{split}.jsonl",encoding="utf-8") as f:
        for line in f:
            r=json.loads(line); r["split"]=split; rows.append(r)
json.dump(rows,open("data/redef_all.json","w"))
print(len(rows),"changes,",sum(r["defective_modification"] for r in rows),"defective")
