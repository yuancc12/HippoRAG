import os
import json
import datetime
import logging
import argparse

from dotenv import load_dotenv
load_dotenv(override=True)

from src.hipporag.HippoRAG import HippoRAG
from src.hipporag.utils.misc_utils import string_to_bool
from src.hipporag.utils.config_utils import BaseConfig

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from main import get_gold_docs, get_gold_answers


def main():
    parser = argparse.ArgumentParser(description="HippoRAG IRCoT QA")
    parser.add_argument('--dataset', type=str, default='2wikimultihopqa')
    parser.add_argument('--llm_base_url', type=str, default='https://api.openai.com/v1')
    parser.add_argument('--llm_name', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--embedding_name', type=str, default='text-embedding-3-small')
    parser.add_argument('--max_qa_steps', type=int, default=2, help='Max IRCoT steps (paper: 2 for 2wiki/hotpotqa, 4 for musique)')
    parser.add_argument('--llm_request_delay', type=float, default=0.5, help='Seconds to sleep after each non-cached LLM API call, to avoid HTTP 429 rate-limit errors')
    parser.add_argument('--checkpoint_every', type=int, default=10, help='Save an IRCoT checkpoint every N completed queries')
    parser.add_argument('--force_index_from_scratch', type=str, default='false')
    parser.add_argument('--force_openie_from_scratch', type=str, default='false')
    parser.add_argument('--openie_mode', choices=['online', 'offline'], default='online')
    parser.add_argument('--save_dir', type=str, default='outputs')
    args = parser.parse_args()

    dataset_name = args.dataset
    save_dir = args.save_dir
    if save_dir == 'outputs':
        save_dir = f'outputs/{dataset_name}'
    else:
        save_dir = f'{save_dir}_{dataset_name}'

    corpus_path = f"reproduce/dataset/{dataset_name}_corpus.json"
    with open(corpus_path, "r") as f:
        corpus = json.load(f)
    docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]

    samples = json.load(open(f"reproduce/dataset/{dataset_name}.json", "r"))
    all_queries = [s['question'] for s in samples]
    gold_answers = get_gold_answers(samples)
    try:
        gold_docs = get_gold_docs(samples, dataset_name)
        assert len(all_queries) == len(gold_docs) == len(gold_answers)
    except Exception:
        gold_docs = None

    config = BaseConfig(
        save_dir=save_dir,
        llm_base_url=args.llm_base_url,
        llm_name=args.llm_name,
        dataset=dataset_name,
        embedding_model_name=args.embedding_name,
        force_index_from_scratch=string_to_bool(args.force_index_from_scratch),
        force_openie_from_scratch=string_to_bool(args.force_openie_from_scratch),
        llm_request_delay=args.llm_request_delay,
        rerank_dspy_file_path="src/hipporag/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json",
        retrieval_top_k=200,
        linking_top_k=5,
        max_qa_steps=args.max_qa_steps,
        qa_top_k=10,
        graph_type="facts_and_sim_passage_node_unidirectional",
        embedding_batch_size=8,
        max_new_tokens=None,
        corpus_len=len(corpus),
        openie_mode=args.openie_mode,
    )

    logging.basicConfig(level=logging.INFO)

    hipporag = HippoRAG(global_config=config)
    hipporag.index(docs)

    checkpoint_path = os.path.join(save_dir, f"ircot_checkpoint_{args.llm_name.replace('/', '_')}.json")
    results = hipporag.ircot_qa(
        queries=all_queries,
        gold_docs=gold_docs,
        gold_answers=gold_answers,
        checkpoint_path=checkpoint_path,
        checkpoint_every=args.checkpoint_every,
    )

    if results is not None and len(results) == 5:
        _, _, _, retrieval_result, qa_result = results
        summary = {
            "dataset": dataset_name,
            "method": "IRCoT+HippoRAG",
            "llm": args.llm_name,
            "embedding": args.embedding_name,
            "max_qa_steps": args.max_qa_steps,
            "retrieval": retrieval_result,
            "qa": qa_result,
        }
        result_path = os.path.join(save_dir, f"ircot_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(result_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logging.info(f"IRCoT results saved to {result_path}")


if __name__ == "__main__":
    main()
