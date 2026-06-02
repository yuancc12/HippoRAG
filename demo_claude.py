"""
Demo: HippoRAG with Anthropic Claude LLM + text-embedding-3-small (OpenAI)
Requires: ANTHROPIC_API_KEY and OPENAI_API_KEY set in .env or environment.
"""
import os
from dotenv import load_dotenv

load_dotenv()



def main():
    docs = [
        "Oliver Badman is a politician.",
        "George Rankin is a politician.",
        "Thomas Marwick is a politician.",
        "Cinderella attended the royal ball.",
        "The prince used the lost glass slipper to search the kingdom.",
        "When the slipper fit perfectly, Cinderella was reunited with the prince.",
        "Erik Hort's birthplace is Montebello.",
        "Marina is born in Minsk.",
        "Montebello is a part of Rockland County.",
    ]

    save_dir = "outputs/claude"
    llm_model_name = "claude-sonnet-4-6"        # or claude-opus-4-7 / claude-haiku-4-5-20251001
    embedding_model_name = "Transformers/sentence-transformers/all-MiniLM-L6-v2"  # local, no API key needed

    hipporag = HippoRAG(
        save_dir=save_dir,
        llm_model_name=llm_model_name,
        embedding_model_name=embedding_model_name,
    )

    hipporag.index(docs=docs)

    queries = [
        "What is George Rankin's occupation?",
        "How did Cinderella reach her happy ending?",
        "What county is Erik Hort's birthplace a part of?",
    ]

    answers = [
        ["Politician"],
        ["By going to the ball."],
        ["Rockland County"],
    ]

    gold_docs = [
        ["George Rankin is a politician."],
        [
            "Cinderella attended the royal ball.",
            "The prince used the lost glass slipper to search the kingdom.",
            "When the slipper fit perfectly, Cinderella was reunited with the prince.",
        ],
        ["Erik Hort's birthplace is Montebello.", "Montebello is a part of Rockland County."],
    ]

    result = hipporag.rag_qa(queries=queries, gold_docs=gold_docs, gold_answers=answers)
    print(result)


if __name__ == "__main__":
    from src.hipporag import HippoRAG

    main()
