"""
evaluate.py LLM-as-a-Judge evaluation. Gemma-compatible (no system_instruction).
Handles rate limits with exponential backoff and skips gracefully on failure.

Usage:
    set GEMINI_API_KEY=your_key   (Windows)
    python evaluate.py
"""
from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from rag_pipeline import RAGPipeline

#  Config 
API_KEY     = os.environ.get("GEMINI_API_KEY", "")
JUDGE_MODEL = "gemma-3-27b-it"   # 30 RPM free tier, no system_instruction
GOLDEN_FILE = "golden_dataset.json"
OUTPUT_FILE = "test_results.json"

RETRIEVAL_N         = 5
RELEVANCE_THRESHOLD = 0.35

# Pacing — Gemma allows 30 RPM but we make 2 calls per question (chatbot + judge)
# so 3s between calls keeps us safely under 20 RPM total
BETWEEN_CALLS_SLEEP = 4    # seconds between every API call
MAX_RETRIES         = 4
BACKOFF_BASE        = 20   # seconds; doubles each retry: 20 → 40 → 80 → 160


# Retry wrapper
def _call_with_retry(fn, *args, **kwargs):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except ResourceExhausted as e:
            wait = BACKOFF_BASE * (2 ** (attempt - 1))
            m = re.search(r"retry in ([0-9.]+)s", str(e))
            if m:
                wait = max(wait, float(m.group(1)) + 3)
            if attempt == MAX_RETRIES:
                print(f"  ✗ Rate limit – giving up after {MAX_RETRIES} attempts.")
                raise
            print(f"  ⏳ Rate limit (attempt {attempt}/{MAX_RETRIES}) – waiting {wait:.0f}s…")
            time.sleep(wait)
        except Exception as e:
            raise RuntimeError(f"API error: {e}") from e


# Judge (Gemma-compatible: system prompt baked into first user turn)
JUDGE_SYSTEM_TURN = """You are an impartial evaluation judge for a RAG chatbot.
You will receive a question, a reference answer, the chatbot's answer, and whether
the question is a "negative example" (meaning the answer does NOT exist in the knowledge base).

Respond with ONLY valid JSON — no markdown fences, no prose — in exactly this schema:
{
  "correctness":  <integer 0-3>,
  "faithfulness": <integer 0-3>,
  "reasoning":    "<one sentence>",
  "verdict":      "<PASS | PARTIAL | FAIL>"
}

Scoring:
  correctness  (vs reference answer): 3=fully correct, 2=mostly correct, 1=partial, 0=wrong/hallucinated
  faithfulness (no hallucination):    3=fully grounded, 2=mostly grounded, 1=significant gaps, 0=hallucinated
  verdict: PASS if both>=2, PARTIAL if either>=1 but not both>=2, FAIL if either==0

For NEGATIVE examples:
  - Chatbot refused / said it doesn't know → correctness=3, faithfulness=3, verdict=PASS
  - Chatbot hallucinated an answer         → correctness=0, faithfulness=0, verdict=FAIL

Confirm you understand with exactly: {"understood": true}""".strip()


def build_judge_model() -> genai.GenerativeModel:
    """Create the Gemma judge model with the system prompt baked into history."""
    genai.configure(api_key=API_KEY)
    return genai.GenerativeModel(model_name=JUDGE_MODEL)


def judge_answer(
    question: str,
    expected: str,
    chatbot_answer: str,
    is_negative: bool,
    judge_model: genai.GenerativeModel,
) -> dict:
    """Run the LLM judge on a single Q&A pair."""

    # Gemma: simulate system prompt via a priming exchange
    history = [
        {"role": "user",  "parts": [JUDGE_SYSTEM_TURN]},
        {"role": "model", "parts": ['{"understood": true}']},
    ]

    user_msg = (
        f"Question: {question}\n\n"
        f"Reference answer: {expected}\n\n"
        f"Chatbot answer: {chatbot_answer}\n\n"
        f"Is negative example: {is_negative}"
    )

    chat     = judge_model.start_chat(history=history)
    response = _call_with_retry(chat.send_message, user_msg)
    raw      = response.text.strip().replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "correctness":  -1,
            "faithfulness": -1,
            "reasoning":    f"Parse error. Raw: {raw[:300]}",
            "verdict":      "ERROR",
        }


# Main
def main() -> None:
    if not API_KEY:
        raise ValueError(
            "GEMINI_API_KEY is not set.\n"
            "Windows : set GEMINI_API_KEY=your_key\n"
            "Mac/Linux: export GEMINI_API_KEY=your_key"
        )

    print("Loading RAG pipeline…")
    pipeline = RAGPipeline(chroma_path="chroma_db", gitlab_path="Gitlab")

    count = pipeline._collection.count()
    if count == 0:
        print("ChromaDB empty – ingesting…")
        n = pipeline.ingest()
        print(f"  Ingested {n} chunks.")
    else:
        print(f"  ChromaDB has {count} chunks.")

    judge_model = build_judge_model()

    with open(GOLDEN_FILE, "r", encoding="utf-8") as f:
        golden = json.load(f)

    total = len(golden)
    print(f"\nEvaluating {total} questions with {JUDGE_MODEL}.")
    print(f"Pacing: {BETWEEN_CALLS_SLEEP}s between API calls.\n")

    results     = []
    total_corr  = 0
    total_faith = 0
    pass_count  = 0
    error_count = 0

    for idx, item in enumerate(golden, 1):
        qid         = item["id"]
        question    = item["question"]
        expected    = item["expected_answer"]
        is_negative = item.get("is_negative", False)

        print(f"[{idx}/{total}] {question[:75]}{'…' if len(question)>75 else ''}")

        # Step 1: get chatbot answer
        chatbot_answer, sources = "ERROR", []
        try:
            chatbot_answer, sources = _call_with_retry(
                pipeline.query,
                question=question,
                api_key=API_KEY,
                n_results=RETRIEVAL_N,
                relevance_threshold=RELEVANCE_THRESHOLD,
            )
            print(f"  Chatbot : {chatbot_answer[:100]}{'…' if len(chatbot_answer)>100 else ''}")
            print(f"  Sources : {sources}")
        except Exception as e:
            print(f"  ✗ Chatbot call failed: {e}")
            error_count += 1
            results.append({
                "id": qid, "question": question, "expected": expected,
                "is_negative": is_negative, "chatbot_answer": "FAILED",
                "sources": [], "scores": {"verdict": "ERROR", "reasoning": str(e)},
            })
            time.sleep(BETWEEN_CALLS_SLEEP)
            continue

        # Pace before judge call 
        time.sleep(BETWEEN_CALLS_SLEEP)

        # Step 2: LLM judge
        scores = {"correctness": -1, "faithfulness": -1, "reasoning": "Skipped", "verdict": "ERROR"}
        try:
            scores = judge_answer(question, expected, chatbot_answer, is_negative, judge_model)
        except Exception as e:
            print(f"  ✗ Judge call failed: {e}")
            error_count += 1

        print(f"  Judge   : correctness={scores.get('correctness')}  "
              f"faithfulness={scores.get('faithfulness')}  "
              f"verdict={scores.get('verdict')}")
        print(f"  Reason  : {scores.get('reasoning', '')}\n")

        corr  = scores.get("correctness",  -1)
        faith = scores.get("faithfulness", -1)
        if isinstance(corr,  int) and corr  >= 0: total_corr  += corr
        if isinstance(faith, int) and faith >= 0: total_faith += faith
        if scores.get("verdict") == "PASS":  pass_count  += 1
        if scores.get("verdict") == "ERROR": error_count += 1

        results.append({
            "id": qid, "question": question, "expected": expected,
            "is_negative": is_negative, "chatbot_answer": chatbot_answer,
            "sources": sources, "scores": scores,
        })

        # Pace before next question
        if idx < total:
            time.sleep(BETWEEN_CALLS_SLEEP)

    #  Summary 
    valid_n = total - error_count
    summary = {
        "run_timestamp":        datetime.now().isoformat(),
        "model_used":           JUDGE_MODEL,
        "total_questions":      total,
        "pass_count":           pass_count,
        "partial_or_fail":      total - pass_count - error_count,
        "error_count":          error_count,
        "pass_rate_pct":        round(pass_count / total * 100, 1),
        "avg_correctness":      round(total_corr  / valid_n, 2) if valid_n else 0,
        "avg_faithfulness":     round(total_faith / valid_n, 2) if valid_n else 0,
        "max_score_per_metric": 3,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    print("=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Model              : {JUDGE_MODEL}")
    print(f"  Total questions    : {total}")
    print(f"  PASS               : {pass_count}  ({summary['pass_rate_pct']}%)")
    print(f"  PARTIAL / FAIL     : {summary['partial_or_fail']}")
    print(f"  Errors / Skipped   : {error_count}")
    print(f"  Avg Correctness    : {summary['avg_correctness']} / 3")
    print(f"  Avg Faithfulness   : {summary['avg_faithfulness']} / 3")
    print(f"\n  Results saved → {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
