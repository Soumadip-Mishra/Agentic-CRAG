"""
Ragas evaluation script.

Runs Faithfulness and Answer Relevance metrics against
the ground-truth QA benchmark dataset.

Usage:
    python -m eval.eval_ragas
    python eval/eval_ragas.py
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

# Suppress Ragas internal migration deprecation notices
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv

# ── Bootstrap: project root on sys.path & .env loaded ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

from datasets import Dataset
from langchain_openai import OpenAIEmbeddings
from loguru import logger
from openai import AsyncOpenAI, OpenAI
from ragas import evaluate
from ragas.llms import llm_factory
from ragas.metrics import answer_relevancy, faithfulness

from config.settings import get_settings
from src.agent.workflow import compile_workflow

# ── Export API key to os.environ for any library that needs it ──
_settings = get_settings()
if _settings.openai_api_key:
    os.environ["OPENAI_API_KEY"] = _settings.openai_api_key

EVAL_DATASET_PATH = Path("data/eval_dataset.json")


def load_eval_dataset() -> list[dict]:
    """Load the ground-truth QA evaluation dataset."""
    with open(EVAL_DATASET_PATH) as f:
        return json.load(f)


def run_evaluation() -> dict:
    """
    Execute the CRAG agent on each eval question,
    then score with Ragas metrics.
    """
    settings = get_settings()
    logger.info("═══ Ragas Evaluation Started ═══")

    qa_pairs = load_eval_dataset()
    agent = compile_workflow()

    questions = []
    answers = []
    contexts_list = []
    ground_truths = []

    for qa in qa_pairs:
        question = qa["question"]
        logger.info(f"Evaluating: {question}")

        result = agent.invoke({
            "question": question,
            "rewritten_question": "",
            "documents": [],
            "web_results": [],
            "relevance_scores": [],
            "relevance_decision": "",
            "generation": "",
            "hallucination_score": 0.0,
            "retry_count": 0,
            "route": "",
        })

        questions.append(question)
        answers.append(result.get("generation", ""))
        contexts_list.append([doc.page_content for doc in result.get("documents", [])])
        ground_truths.append(qa["ground_truth"])

    # ── Build Ragas dataset ──
    ragas_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    })

    # ── Configure LLM and Embeddings for Ragas evaluation ──
    from ragas.run_config import RunConfig

    openai_client = OpenAI(api_key=settings.openai_api_key)
    ragas_llm = llm_factory(
        model="gpt-4o-mini",
        client=openai_client,
        max_tokens=8192,
    )
    embedder = OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key,
    )
    run_config = RunConfig(
        timeout=180,
        max_retries=3,
        max_wait=30,
        max_workers=1,
    )

    # ── Evaluate ──
    results = evaluate(
        ragas_dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=ragas_llm,
        embeddings=embedder,
        run_config=run_config,
    )

    logger.info(f"═══ Ragas Results ═══\n{results}")
    return results


if __name__ == "__main__":
    run_evaluation()
