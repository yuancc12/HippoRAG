"""
Demo: HippoRAG with Anthropic Claude LLM + local sentence-transformers embedding
Requires: ANTHROPIC_API_KEY set in .env
Optional: OPENAI_API_KEY for text-embedding-3-small
"""
import os
import logging
from dotenv import load_dotenv

load_dotenv()

# Enable INFO logging so the execution flow is visible
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def visualize_graph(hipporag, save_path="outputs/claude/graph.png"):
    """Visualize the HippoRAG knowledge graph and save to file."""
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

        # Shorten long labels
        short_labels = [l[:35] + "…" if len(l) > 35 else l for l in labels]

        # Color by node type derived from hash_id prefix
        def node_color(hid):
            if hid.startswith("entity-"):
                return "#4C9BE8"   # blue = entity
            elif hid.startswith("chunk-"):
                return "#F4A261"   # orange = passage/chunk
            return "#aaaaaa"

        colors = [node_color(h) for h in hash_ids]
        sizes = [15 if h.startswith("entity-") else 25 for h in hash_ids]

        fig, ax = plt.subplots(figsize=(16, 10))
        layout = g.layout("fr")  # Fruchterman-Reingold

        ig.plot(
            g,
            target=ax,
            layout=layout,
            vertex_label=short_labels,
            vertex_color=colors,
            vertex_size=sizes,
            vertex_label_size=7,
            edge_arrow_size=0.5,
            edge_color="#cccccc",
        )

        legend = [
            mpatches.Patch(color="#4C9BE8", label="Entity node (small)"),
            mpatches.Patch(color="#F4A261", label="Passage / Chunk node (large)"),
        ]
        ax.legend(handles=legend, loc="upper left", fontsize=8)
        ax.set_title("HippoRAG Knowledge Graph\n(Blue=Entity, Orange=Passage)", fontsize=13)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.show()
        print(f"\nGraph saved to: {save_path}")

    except Exception as e:
        print(f"Graph visualization failed: {e}")


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
    # embedding_model_name = "text-embedding-3-small"  # OpenAI embedding (requires OPENAI_API_KEY with credits)
    embedding_model_name = "Transformers/sentence-transformers/all-MiniLM-L6-v2"  # local, no API key needed

    print("\n" + "="*60)
    print("STEP 1: Initializing HippoRAG")
    print("="*60)
    hipporag = HippoRAG(
        save_dir=save_dir,
        llm_model_name=llm_model_name,
        embedding_model_name=embedding_model_name,
    )

    print("\n" + "="*60)
    print("STEP 2: Indexing documents")
    print(f"  {len(docs)} documents to index")
    print("  - NER: extract named entities via Claude")
    print("  - Triple extraction: build (subject, relation, object) triples via Claude")
    print("  - Embedding: encode chunks, entities, facts")
    print("  - Graph construction: build knowledge graph")
    print("="*60)
    hipporag.index(docs=docs)

    print("\n" + "="*60)
    print("STEP 3: Graph statistics")
    print("="*60)
    g = hipporag.graph
    node_types = {}
    if "type" in g.vertex_attributes():
        for t in g.vs["type"]:
            node_types[t] = node_types.get(t, 0) + 1
    print(f"  Nodes: {g.vcount()}  (breakdown: {node_types})")
    print(f"  Edges: {g.ecount()}")
    print(f"  Node names: {g.vs['name'][:10]}{'...' if g.vcount() > 10 else ''}")

    print("\n" + "="*60)
    print("STEP 4: Visualizing knowledge graph")
    print("="*60)
    visualize_graph(hipporag)

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

    print("\n" + "="*60)
    print("STEP 5: RAG QA")
    print("  - Retrieve relevant passages via graph traversal")
    print("  - Answer questions using Claude")
    print("="*60)
    solutions, raw_answers, token_usages, retrieval_metrics, qa_metrics = hipporag.rag_qa(
        queries=queries, gold_docs=gold_docs, gold_answers=answers
    )

    print("\n" + "="*60)
    print("STEP 6: Results")
    print("="*60)
    for sol, raw in zip(solutions, raw_answers):
        print(f"\nQ: {sol.question}")
        print(f"A: {sol.answer}")
        print(f"   (retrieved {len(sol.docs)} docs, top doc: '{sol.docs[0][:60]}')")

    print(f"\nRetrieval: {retrieval_metrics}")
    print(f"QA:        {qa_metrics}")


if __name__ == "__main__":
    from src.hipporag import HippoRAG

    main()
