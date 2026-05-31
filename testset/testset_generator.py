from langchain_community.document_loaders import DirectoryLoader 
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.llms import Ollama
from ragas.embeddings import LangchainEmbeddingsWrapper
#from langchain_openai import ChatOpenAI
import asyncio
import nest_asyncio
import pandas as pd
from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer
from ragas.testset.persona import Persona

import pymupdf4llm
from pathlib import Path

#nest_asyncio.apply()

OLLAMA_CLOUD_URL = "https://ollama.com"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_BASE_MODEL = "gpt-oss:120b-cloud"
OLLAMA_EMBEDDING_MODEL = "qwen3-embedding:0.6b"
OLLAMA_API_KEY = "da62361e1f074c83a785c95430955175.R7Nu4daE7L7NwvNTq2kcVnuf"

async def generate_testset():
    # llm = ChatOllama(
    #     model=OLLAMA_BASE_MODEL,
    #     base_url=OLLAMA_BASE_URL,
    #     temperature=0,
    # )


    llm = Ollama(
        model=OLLAMA_BASE_MODEL,
        base_url=OLLAMA_CLOUD_URL,
        temperature=0,
        headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"}
    )

    embeddings = OllamaEmbeddings(
        model=OLLAMA_EMBEDDING_MODEL, 
        base_url=OLLAMA_BASE_URL,
    )

    generator_llm = LangchainLLMWrapper(llm)
    generator_embeddings = LangchainEmbeddingsWrapper(embeddings)

    md_text = pymupdf4llm.to_markdown("../documents/CELEX_32006L0054_EN_TXT.pdf")
    Path("output.md").write_bytes(md_text.encode())

    loader = DirectoryLoader('.', glob="output.md")
    docs = loader.load()

    # IT questions
    # italian_synthesizer = SingleHopSpecificQuerySynthesizer(llm=generator_llm)
    # italian_prompts = await italian_synthesizer.adapt_prompts("italian", llm=generator_llm)
    # italian_synthesizer.set_prompts(**italian_prompts)

    # EN questions
    english_synthesizer = SingleHopSpecificQuerySynthesizer(llm=generator_llm)

    query_distribution = [
            (english_synthesizer, 1.0) # 100% of questions will use this Italian workflow
    ]

    generator = TestsetGenerator(
            llm=generator_llm, 
            embedding_model=generator_embeddings
    )

    generator.persona_list = [
        Persona(
            name="HR Compliance Specialist", 
            role_description="A legal expert focused on corporate governance, workplace compliance, and labor laws." #"Un esperto legale incentrato sulla conformità aziendale e sul diritto del lavoro."
        )
    ]

    loop = asyncio.get_running_loop()

    testset = await loop.run_in_executor(
        None,  # Uses default ThreadPoolExecutor
        lambda: generator.generate_with_langchain_docs(
            documents=docs,
            testset_size=11,
            query_distribution=query_distribution
        )
    )


    res = testset.to_pandas()

    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", 1000)

    res.to_csv("testset_generated_en.csv", index=False) 

# async def main():
#     # Force an explicit async task allocation context for sniffio to hook into
#     task = asyncio.create_task(generate_testset())
#     await task

# if __name__ == "__main__":
#     # Ensure the loop is established first, then execute our main task wrapper
#     loop = asyncio.get_event_loop()
#     loop.run_until_complete(main())

if __name__ == "__main__":
    # Standard, clean async entry point. No nest_asyncio hacks required.
    asyncio.run(generate_testset())