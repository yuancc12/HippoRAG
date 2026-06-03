"""
Demo: HippoRAG with Anthropic Claude LLM + local sentence-transformers embedding
Requires: ANTHROPIC_API_KEY set in .env
Optional: OPENAI_API_KEY for text-embedding-3-small

Usage:
  python demo_claude.py              # use built-in toy data
  python demo_claude.py sample       # use reproduce/dataset/sample (1 QA, 3 Wikipedia passages)
  python demo_claude.py hotpotqa     # use reproduce/dataset/hotpotqa (1000 QA, 9811 passages)
  python demo_claude.py musique      # use reproduce/dataset/musique  (1000 QA, 11656 passages)
  python demo_claude.py 2wikimultihopqa
"""
import os
import sys
import json
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


# ── Dataset loader ────────────────────────────────────────────────────────────

def load_dataset(name: str):
    """
    Load corpus + QA pairs from reproduce/dataset/<name>_corpus.json and <name>.json.
    Returns (docs, queries, gold_answers, gold_docs).
    """
    corpus_path = f"reproduce/dataset/{name}_corpus.json"
    qa_path = f"reproduce/dataset/{name}.json"

    with open(corpus_path, encoding="utf-8") as f:
        corpus = json.load(f)
    with open(qa_path, encoding="utf-8") as f:
        qa_data = json.load(f)

    # corpus: [{title, text, idx}, ...]  → docs = "Title\nText"
    docs = [f"{p['title']}\n{p['text']}" for p in corpus]

    queries, gold_answers, gold_docs = [], [], []
    for item in qa_data:
        queries.append(item["question"])
        gold_answers.append(item["answer"] if isinstance(item["answer"], list) else [item["answer"]])
        # paragraphs field contains the gold supporting passages (if available)
        if "paragraphs" in item:
            gold_docs.append([f"{p['title']}\n{p['text']}" for p in item["paragraphs"]])
        else:
            gold_docs.append([])

    print(f"Loaded '{name}': {len(docs)} passages, {len(queries)} questions")
    return docs, queries, gold_answers, gold_docs


def load_toy_data():
    """Built-in toy dataset (no file needed)."""
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
    queries = [
        "What is George Rankin's occupation?",
        "How did Cinderella reach her happy ending?",
        "What county is Erik Hort's birthplace a part of?",
    ]
    gold_answers = [["Politician"], ["By going to the ball."], ["Rockland County"]]
    gold_docs = [
        ["George Rankin is a politician."],
        ["Cinderella attended the royal ball.",
         "The prince used the lost glass slipper to search the kingdom.",
         "When the slipper fit perfectly, Cinderella was reunited with the prince."],
        ["Erik Hort's birthplace is Montebello.", "Montebello is a part of Rockland County."],
    ]
    print("Using built-in toy data (9 passages, 3 questions)")
    return docs, queries, gold_answers, gold_docs


# ── Graph visualization ───────────────────────────────────────────────────────

def visualize_graph(hipporag, save_path):
    try:
        import igraph as ig
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        g = hipporag.graph
        if g is None or g.vcount() == 0:
            print("Graph is empty, skipping visualization.")
            return

        labels = g.vs["content"] if "content" in g.vertex_attributes() else [str(i) for i in range(g.vcount())]
        hash_ids = g.vs["hash_id"] if "hash_id" in g.vertex_attributes() else [""] * g.vcount()
        short_labels = [l[:35] + "..." if len(l) > 35 else l for l in labels]

        def node_color(hid):
            if hid.startswith("entity-"): return "#4C9BE8"
            if hid.startswith("chunk-"):  return "#F4A261"
            return "#aaaaaa"

        colors = [node_color(h) for h in hash_ids]
        sizes  = [15 if h.startswith("entity-") else 25 for h in hash_ids]

        fig, ax = plt.subplots(figsize=(18, 11))
        ig.plot(g, target=ax, layout=g.layout("fr"),
                vertex_label=short_labels, vertex_color=colors,
                vertex_size=sizes, vertex_label_size=7,
                edge_arrow_size=0.5, edge_color="#cccccc")

        ax.legend(handles=[
            mpatches.Patch(color="#4C9BE8", label="Entity node"),
            mpatches.Patch(color="#F4A261", label="Passage / Chunk node"),
        ], loc="upper left", fontsize=8)
        ax.set_title("HippoRAG Knowledge Graph  (Blue=Entity, Orange=Passage)", fontsize=13)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.show()
        print(f"Graph saved to: {save_path}")

    except Exception as e:
        print(f"Graph visualization failed: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else None

    if dataset_name:
        docs, queries, gold_answers, gold_docs = load_dataset(dataset_name)
        save_dir = f"outputs/claude_{dataset_name}"
    else:
        docs, queries, gold_answers, gold_docs = load_toy_data()
        save_dir = "outputs/claude"

    llm_model_name       = "claude-sonnet-4-6"
    # embedding_model_name = "text-embedding-3-small"          # OpenAI (requires credits)
    embedding_model_name = "Transformers/sentence-transformers/all-MiniLM-L6-v2"  # local, free

    print("\n" + "="*60)
    print("STEP 1: Initializing HippoRAG")
    print("="*60)
    hipporag = HippoRAG(
        save_dir=save_dir,
        llm_model_name=llm_model_name,
        embedding_model_name=embedding_model_name,
    )

    print("\n" + "="*60)
    print(f"STEP 2: Indexing {len(docs)} documents")
    print("  NER + Triple extraction (Claude) -> Embedding -> Graph")
    print("="*60)
    hipporag.index(docs=docs)

    print("\n" + "="*60)
    print("STEP 3: Graph statistics")
    print("="*60)
    g = hipporag.graph
    entity_count  = sum(1 for h in g.vs["hash_id"] if h.startswith("entity-"))
    passage_count = sum(1 for h in g.vs["hash_id"] if h.startswith("chunk-"))
    print(f"  Nodes: {g.vcount()}  (entity={entity_count}, passage={passage_count})")
    print(f"  Edges: {g.ecount()}")

    print("\n" + "="*60)
    print("STEP 4: Visualizing knowledge graph")
    print("="*60)
    visualize_graph(hipporag, save_path=f"{save_dir}/graph.png")

    print("\n" + "="*60)
    print(f"STEP 5: RAG QA  ({len(queries)} questions)")
    print("  Graph PPR retrieval -> Claude reads passages -> answer")
    print("="*60)
    solutions, raw_answers, token_usages, retrieval_metrics, qa_metrics = hipporag.rag_qa(
        queries=queries, gold_docs=gold_docs, gold_answers=gold_answers
    )

    print("\n" + "="*60)
    print("STEP 6: Results")
    print("="*60)
    for sol in solutions:
        print(f"\nQ: {sol.question}")
        print(f"A: {sol.answer}")
        print(f"   top doc: '{sol.docs[0][:80]}'")

    print(f"\nRetrieval: {retrieval_metrics}")
    print(f"QA:        {qa_metrics}")


if __name__ == "__main__":
    from src.hipporag import HippoRAG
    main()
