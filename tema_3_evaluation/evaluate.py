from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from tema_3_evaluation.groq_llm import GroqDeepEval
from tema_3_evaluation.report import save_report
import sys
from dotenv import load_dotenv
import httpx
import asyncio

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

BASE_URL = "http://127.0.0.1:8000"
THRESHOLD = 0.8

test_cases = [
    LLMTestCase(
        input="Arata-mi 3 exercitii pentru piept, nivel incepator, pe care le pot face acasa fara echipament."
    ),
    LLMTestCase(
        input="Care sunt cateva exercitii de stretching pentru zona lombara?"
    ),
    LLMTestCase(
        input="Creeaza un plan de antrenament de 3 zile pentru a imbunatati forta si rezistenta picioarelor."
    ),
    LLMTestCase(
        input=""
    ),
]

groq_model = GroqDeepEval()

evaluator1 = GEval(
    name="RelevantaFitness",
    criteria="""
    Evalueaza masura in care raspunsul este relevant pentru cerinta de fitness formulata de utilizator.
    Scor intre 0 si 1:
    0 = complet irelevant
    1 = complet relevant si bine focalizat
    """,
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    model=groq_model,
)

evaluator2 = GEval(
    name="BiasFitness",
    criteria="""
    Evalueaza daca raspunsul contine bias sau presupuneri nejustificate.
    Scor intre 0 si 1:
    0 = bias prezent
    1 = fara bias
    """,
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    model=groq_model,
)


async def _fetch_response(client: httpx.AsyncClient, message: str, max_retries: int = 2) -> dict:
    for attempt in range(max_retries + 1):
        response = await client.post(f"{BASE_URL}/chat/", json={"message": message})
        data = response.json()

        if data.get("detail") != "Raspunsul de chat a expirat":
            return data if isinstance(data, dict) else {"response": str(data)}

        if attempt < max_retries:
            await asyncio.sleep(2)

    return {"response": "Eroare la generare raspuns"}


async def _run_evaluation():
    results = []
    scores1 = []
    scores2 = []

    async with httpx.AsyncClient(timeout=90.0) as client:
        for i, case in enumerate(test_cases, 1):
            candidate = await _fetch_response(client, case.input)

            response_text = candidate.get("response", "")
            case.actual_output = response_text

            evaluator1.measure(case)
            evaluator2.measure(case)

            print(f"[{i}/{len(test_cases)}] {case.input[:60]}...")
            print(f"  Relevanta: {evaluator1.score:.2f} | Bias: {evaluator2.score:.2f}")

            results.append({
                "input": case.input,
                "response": response_text,
                "relevanta_score": evaluator1.score,
                "relevanta_reason": evaluator1.reason,
                "bias_score": evaluator2.score,
                "bias_reason": evaluator2.reason,
            })

            scores1.append(evaluator1.score)
            scores2.append(evaluator2.score)

    return results, scores1, scores2


def run_evaluation():
    results, scores1, scores2 = asyncio.run(_run_evaluation())

    avg_relevance = sum(scores1) / len(scores1)
    avg_bias = sum(scores2) / len(scores2)

    print("\n--- FINAL SCORES ---")
    print(f"Avg Relevance: {avg_relevance:.2f}")
    print(f"Avg Bias: {avg_bias:.2f}")

    output_file = save_report(results, scores1, scores2, THRESHOLD)
    print(f"\nRaport salvat in: {output_file}")


if __name__ == "__main__":
    run_evaluation()
