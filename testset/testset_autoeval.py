from typing import List, Literal
from pydantic import BaseModel, Field
import json
import pandas as pd
from pathlib import Path
import pymupdf4llm
from langchain_text_splitters import MarkdownTextSplitter
from openai import OpenAI
import instructor

class QuestionStructure(BaseModel):
    query_type: Literal["factual", "reasoning", "conditional", "comparative"] = Field(
        description="The structural type of the question. 'factual' for direct retrieval, 'reasoning' for multi-step logic, 'conditional' for if/then scenarios, 'comparative' for comparing articles."
    )
    difficulty: Literal["easy", "medium", "hard"] = Field(
        description="The cognitive complexity: 'easy' (surface facts), 'medium' (requires interpretation), 'hard' (highly complex legal analysis)."
    )
    language: str = Field(description="The language used for this specific item (e.g., 'Italian' or 'English')")
    question: str = Field(description="The synthetic evaluation question formulated exactly in the requested language.")
    ground_truth: str = Field(description="The comprehensive, precise answer to the question extracted solely from the context chunk, written in the requested language.")

class EvaluationTestSet(BaseModel):
    items: List[QuestionStructure]

def generate_auto_evaluation_set(
        pdf_path: str,
        output_csv: str = "auto_eval_testset.csv",
        language: str = "Italian",
        num_questions_per_chunk: int = 1
):
    client = instructor.from_openai(
        OpenAI(
            base_url="http://127.0.0.1:11434/v1",
            api_key="ollama"
        ),
        mode=instructor.Mode.JSON
    )

    md_text = pymupdf4llm.to_markdown(pdf_path)
    splitter = MarkdownTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = splitter.split_text(md_text)

    all_generated_items = []

    system_prompt = (
        f"You are an expert AI QA Test Set Generator for advanced RAG evaluations.\n"
        f"Your task is to analyze the provided document excerpt and generate exactly {num_questions_per_chunk} distinct evaluation items.\n"
        f"CRITICAL REQUIREMENTS:\n"
        f"1. You MUST write all questions and ground_truth answers strictly in {language}.\n"
        f"2. Vary the 'query_type' and 'difficulty' dynamically across the items you create.\n"
        f"3. Do not assume facts outside the text excerpt provided."
    )

    for i, chunk in enumerate(chunks[:5]):
        try:
            test_set_response: EvaluationTestSet = client.chat.completions.create(
                model="gpt-oss:120b-cloud",
                response_model=EvaluationTestSet,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context Document Excerpt:\n{chunk}"}
                ],
                temperature=0.4
            )

            for item in test_set_response.items:
                item_dict = item.model_dump()
                item_dict["source_context"] = chunk # Attach source context for validation
                all_generated_items.append(item_dict)

        except Exception as e:
            print(e)
            continue

    if all_generated_items:
        df = pd.DataFrame(all_generated_items)
        df.to_csv(output_csv, index=False)

        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_colwidth", 40)

if __name__ == "__main__":
    generate_auto_evaluation_set(
        pdf_path="../documents/CELEX_32006L0054_IT_TXT.pdf",
        output_csv="italian_legal_testset.csv",
        language="Italian"
    )