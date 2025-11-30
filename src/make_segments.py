
from pathlib import Path
import json
import spacy

RAW_DIR = Path("data/raw")


OUT_FILE = Path("data/annotations/candidate_segments.jsonl")

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2_500_000 



def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out = OUT_FILE.open("w", encoding="utf-8")

    seg_id = 1

    for txt_path in RAW_DIR.glob("*.txt"):
        print(f"Processing {txt_path.name}...")
        text = txt_path.read_text(encoding="utf-8", errors="ignore")

    
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

   
        chunk_size = 3

        for i in range(0, len(sentences), chunk_size):
            chunk_sents = sentences[i:i + chunk_size]
            if len(chunk_sents) < 2:
                continue  # skip too-short chunks

            segment_text = " ".join(chunk_sents)

            obj = {
                "id": f"cand_{seg_id:05d}",
                "text": segment_text,
                "labels": []  # empty for now, you will fill this manually
            }
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            seg_id += 1

    out.close()
    print(f"Saved {seg_id-1} candidate segments to {OUT_FILE}")


if __name__ == "__main__":
    main()
