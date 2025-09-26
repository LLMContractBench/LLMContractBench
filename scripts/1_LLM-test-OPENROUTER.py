    #Written by: Norbert Tihanyi
    import os, json, re, time, csv
    from typing import List, Dict, Any, Optional, Tuple
    from collections import Counter
    from tqdm import tqdm
    from openai import OpenAI, APIError, RateLimitError, InternalServerError, BadRequestError
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # ================== CONFIG ==================
    API_KEY        = "<YOUR-API-KEY-HERE>" 
    MODEL_NAME     = "google/gemini-2.5-pro"
    INPUT_PATH     = "SWC50-dataset.json"
    OUTPUT_DIR     = "./results"
    TEMPERATURE    = 0.2
    MAX_RETRIES    = 4
    SLEEP_BACKOFF  = 0.0
    SHOW_CODE      = False
    TEST_MODE_LIMIT = None
    NUM_WORKERS    = 10   # parallel requests
    # ============================================

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=API_KEY)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"[*] Model: {MODEL_NAME}")
    print(f"[*] Input: {INPUT_PATH}")

    # ================== PROMPT ==================
    GENERAL_PROMPT = """
    You are an expert Ethereum smart contract security auditor.

    I will provide Solidity code. For each contract:

    1. Analyze the code carefully.
    2. Identify vulnerabilities ONLY from the list below.
    3. Respond STRICTLY with one XML tag in the format:

    <XML>TYPE1,TYPE2,...</XML>

    ### Rules:
    - Use ONLY the labels provided below.
    - If multiple vulnerabilities are found, separate them with commas.
    - Do not add explanations, reasoning, or extra text outside the XML tag.
    - Only respond with labels from this exact set: ARTHM,DOS,LE,RENT,TM,TOO,TXO,UE.
    ### Vulnerability types:

    - ARTHM = Arithmetic errors (overflow, underflow, division by zero)
    - DOS   = Denial of Service (blocking functions or consuming gas)
    - LE    = Logical Errors (incorrect business logic, unexpected behavior)
    - RENT  = Reentrancy (recursive calls before state update)
    - TM    = Time manipulation (insecure use of block.timestamp)
    - TOO   = Incorrect use of time (wrong time dependency for logic)
    - TXO   = Insecure use of tx.origin (phishing/authentication risk)
    - UE    = Unchecked external calls (no verification of call success)


    ### Solidity contract for analysis:
    """

    ALLOWED_LABELS = {"arthm", "dos", "le", "rent", "tm", "too", "txo", "ue"}
    XML_RE = re.compile(r"<xml>.*?</xml>", re.IGNORECASE | re.DOTALL)

    # ================== HELPERS ==================
    def build_prompt(code: str) -> str:
        return f"{GENERAL_PROMPT}\n---\n{code}\n---"

    def extract_xml_answer(text: str) -> Optional[str]:
        """Extract <XML>...</XML> no matter if wrapped in ```xml fences or plain text."""
        if not text:
            return None

        # Remove markdown code fences (```xml ... ```)
        cleaned = re.sub(r"^```[a-zA-Z0-9]*\s*", "", text.strip(), flags=re.IGNORECASE | re.MULTILINE)
        cleaned = re.sub(r"```$", "", cleaned.strip(), flags=re.MULTILINE)

        # Find <XML>...</XML> ignoring case
        m = re.search(r"<xml>.*?</xml>", cleaned, re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(0).strip()
        return None

    def sanitize_xml(xml_text: Optional[str]) -> Optional[str]:
        if not xml_text: return None
        xml_text = xml_text.strip()
        if xml_text.lower().startswith("<xml>") and xml_text.lower().endswith("</xml>"):
            return xml_text
        inner = re.sub(r"[^A-Za-z0-9_,]", "", xml_text).strip()
        return f"<xml>{inner}</xml>" if inner else None

    def parse_labels_from_xml(xml_text: str) -> List[str]:
        content = xml_text.strip()[5:-6].strip().lower()
        if not content or content == "none": return []
        parts = [p.strip() for p in content.split(",")]
        return [p for p in parts if p in ALLOWED_LABELS]

    def parse_ground_truth_labels(rec: Dict[str, Any]) -> List[str]:
        mapping = {
            "ARTHM": "arthm", "DOS": "dos", "LE": "le", "RENT": "rent",
            "TimeM": "tm", "TimeO": "too", "Tx-Origin": "txo", "UE": "ue",
        }
        return [norm for key, norm in mapping.items() if str(rec.get(key, 0)) == "1"]

    def ask_llm(code: str) -> Tuple[Optional[str], Optional[str], str]:
        prompt = build_prompt(code)
        backoff = SLEEP_BACKOFF
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=TEMPERATURE,
                )

                content = resp.choices[0].message.content if resp.choices else ""
                xml = extract_xml_answer(content)
                if xml:
                    xml = sanitize_xml(xml)
                if not xml:
                    return None, "No valid <xml>...</xml>", content
                return xml, None, content
            except (RateLimitError, InternalServerError):
                if attempt == MAX_RETRIES: return None, "Rate/Server error", ""
                time.sleep(backoff); backoff *= 1.7
            except (APIError, BadRequestError) as e:
                return None, f"{type(e).__name__}: {e}", ""
            except Exception as e:
                if attempt == MAX_RETRIES: return None, f"UnexpectedError: {e}", ""
                time.sleep(backoff); backoff *= 1.7

    def process_record(rec: Dict[str, Any]) -> Dict[str, Any]:
        code = rec.get("SMART_CONTRACT_CODE", "")
        if not isinstance(code, str) or not code.strip():
            xml, err, raw = "<xml>NONE</xml>", "Missing SMART_CONTRACT_CODE", ""
            pred_labels = []
        else:
            xml, err, raw = ask_llm(code)
            pred_labels = parse_labels_from_xml(xml) if xml else []

        return {
            "id": rec.get("id"),
            "address": rec.get("addr4sss"),
            "xml": xml or "",
            "pred_labels": pred_labels,
            "gt_labels": parse_ground_truth_labels(rec),
            "error": err,
            "SMART_CONTRACT_CODE": code if SHOW_CODE else None,
        }

    # ================== MAIN ==================
    def main():
        with open(INPUT_PATH, "r", encoding="utf-8") as f:
            records = json.load(f)
        if not isinstance(records, list): records = [records]
        if TEST_MODE_LIMIT: records = records[:TEST_MODE_LIMIT]

        enriched = []
        tp, fp, fn = Counter(), Counter(), Counter()
        num_exact = num_hit_incl_none = num_hit_nonempty = num_done = 0

        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {executor.submit(process_record, rec): rec for rec in records}
            with tqdm(total=len(futures), desc="Auditing", ncols=120) as pbar:
                for fut in as_completed(futures):
                    rec = fut.result()
                    enriched.append(rec)

                    # Update stats live
                    pred_set, gt_set = set(rec["pred_labels"]), set(rec["gt_labels"])
                    num_done += 1
                    if pred_set == gt_set: num_exact += 1
                    inter = pred_set & gt_set
                    if inter or (not pred_set and not gt_set): num_hit_incl_none += 1
                    if inter: num_hit_nonempty += 1
                    for l in gt_set:
                        if l not in pred_set: fn[l] += 1
                    for l in pred_set:
                        if l not in gt_set: fp[l] += 1
                    for l in inter: tp[l] += 1

                    acc = num_exact / num_done * 100
                    hit = num_hit_incl_none / num_done * 100
                    pbar.set_postfix({"Acc": f"{acc:.1f}%", "Hit≥1": f"{hit:.1f}%"}, refresh=True)
                    pbar.update(1)

        # ================== METRICS ==================
        total = len(enriched)
        print("\n Finished.")
        print(f"Exact-match accuracy (EMA): {num_exact/total:.3f}")
        print(f"At least one correct      : {num_hit_nonempty/total:.3f}")

        print("\nBy-label metrics:")
        all_labels = sorted(set(tp) | set(fp) | set(fn))
        for l in all_labels:
            P = tp[l] / (tp[l] + fp[l]) if (tp[l] + fp[l]) else 0
            R = tp[l] / (tp[l] + fn[l]) if (tp[l] + fn[l]) else 0
            F1 = 2*P*R/(P+R) if (P+R) else 0
            print(f" - {l.upper():6s}: P={P:.2f}, R={R:.2f}, F1={F1:.2f}, TP={tp[l]}, FP={fp[l]}, FN={fn[l]}")
            # --- Global Macro/Micro Metrics ---
        
        if all_labels:
            precisions, recalls, f1s = [], [], []
            for l in all_labels:
                P = tp[l] / (tp[l] + fp[l]) if (tp[l] + fp[l]) else 0
                R = tp[l] / (tp[l] + fn[l]) if (tp[l] + fn[l]) else 0
                F1 = 2*P*R/(P+R) if (P+R) else 0
                precisions.append(P); recalls.append(R); f1s.append(F1)

            macro_P = sum(precisions) / len(all_labels)
            macro_R = sum(recalls) / len(all_labels)
            macro_F1 = sum(f1s) / len(all_labels)

            total_TP = sum(tp.values())
            total_FP = sum(fp.values())
            total_FN = sum(fn.values())
            micro_P = total_TP / (total_TP + total_FP) if (total_TP + total_FP) else 0
            micro_R = total_TP / (total_TP + total_FN) if (total_TP + total_FN) else 0
            micro_F1 = 2*micro_P*micro_R/(micro_P+micro_R) if (micro_P+micro_R) else 0

            print("\nGlobal Metrics:")
            print(f" - Macro Avg : P={macro_P:.2f}, R={macro_R:.2f}, F1={macro_F1:.2f}")
            print(f" - Micro Avg : P={micro_P:.2f}, R={micro_R:.2f}, F1={micro_F1:.2f}")

        # Save results
        safe_model = MODEL_NAME.replace("/", "_").replace(":", "_")
        json_path = os.path.join(OUTPUT_DIR, f"{safe_model}_results.json")
        csv_path  = os.path.join(OUTPUT_DIR, f"{safe_model}_results.csv")
        with open(json_path, "w", encoding="utf-8") as f: json.dump(enriched, f, indent=2, ensure_ascii=False)
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id","address","pred_labels","gt_labels","xml","error"])
            for r in enriched:
                writer.writerow([r["id"],r["address"],",".join(r["pred_labels"]),",".join(r["gt_labels"]),r["xml"],r["error"] or ""])
        print("\n [*] Wrote files:")
        print(" -", json_path)
        print(" -", csv_path)

    if __name__ == "__main__":
        main()
