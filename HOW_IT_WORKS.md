# HippoRAG with Claude — 執行流程說明

## 快速啟動

```powershell
# 1. 啟動虛擬環境
.\hipporag_env\Scripts\Activate.ps1

# 2. 執行 demo
python demo_claude.py
```

---

## 完整執行流程

```
【輸入】
  docs    = ["文件1", "文件2", ...]   → 你要建索引的文章/段落
  queries = ["問題1", "問題2", ...]   → 你想問的問題
  （兩者都是 Python list，在 demo_claude.py 裡手動定義）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  STEP 1  INDEX（建圖，只在第一次或文件改變時執行）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  docs
   │
   ├─► Chunking
   │     預設：每筆 doc = 1 chunk（不拆）
   │     可設定：preprocess_chunk_max_token_size=512
   │             preprocess_chunk_overlap_token_size=128
   │
   ├─► [Claude] NER — 從每個 chunk 抽出命名實體
   │     例："Erik Hort's birthplace is Montebello."
   │          → entities: ["Erik Hort", "Montebello"]
   │
   ├─► [Claude] Triple Extraction — 抽三元組
   │     例："Montebello is a part of Rockland County."
   │          → (Montebello, is part of, Rockland County)
   │
   ├─► [sentence-transformers] Embedding — 三類向量各自編碼
   │     chunk 向量    → vdb_chunk.parquet
   │     entity 向量   → vdb_entity.parquet
   │     fact 向量     → vdb_fact.parquet
   │
   └─► Graph Construction — 建知識圖譜
         節點：Entity node（藍）＋ Passage/Chunk node（橘）
         邊：  三元組邊（entity↔entity）
               提及邊（passage → 它包含的 entity）
         儲存：graph.pickle

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  STEP 2  RAG QA（每次查詢都執行）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  query
   │
   ├─► [sentence-transformers] 編碼查詢向量
   │
   ├─► 向量相似度搜尋
   │     → 找最相近的 entity 和 fact
   │
   ├─► PPR（Personalized PageRank）圖譜走訪
   │     從相關 entity 出發，沿邊擴散
   │     → 找到相關的 Passage 節點
   │     （這步讓跨段落的多跳推理成為可能）
   │
   └─► [Claude] 閱讀取回的段落，生成答案

【輸出】
  answer + retrieved_docs + retrieval_metrics + qa_metrics
```

---

## 為什麼需要 Entity？（多跳推理）

只用 Passage 的問題：

```
Q: "What county is Erik Hort's birthplace a part of?"

傳統 RAG（向量相似度）：
  → 只找到 "Erik Hort's birthplace is Montebello."
  → 不知道 Montebello 屬於哪個郡 ✗

HippoRAG（走圖譜）：
  query → entity: "erik hort"
        → entity: "montebello"        （三元組邊）
        → passage: "Montebello is a part of Rockland County."  （提及邊）
  → 兩段合併，Claude 回答 "Rockland County" ✓
```

Entity 的作用：在不同 Passage 之間架橋，讓推理可以跨段落串聯。

---

## 哪些 AI 被使用

| 階段 | 用途 | 模型 |
|------|------|------|
| Index — NER | 從文件抽實體 | Claude（Anthropic API） |
| Index — Triple | 抽三元組 | Claude（Anthropic API） |
| Index — Embedding | 向量編碼 | sentence-transformers（本地） |
| QA — Embedding | 查詢向量編碼 | sentence-transformers（本地） |
| QA — 回答 | 閱讀段落生成答案 | Claude（Anthropic API） |
| 圖譜走訪 PPR | 純數學計算 | 無 AI |

---

## Cache 機制（不需重複呼叫 API）

第二次以後執行 `index()` 時，如果文件沒有改變，全部從 cache 讀取：

```
outputs/claude/
├── openie_results_ner_claude-sonnet-4-6.json   ← NER + 三元組結果
├── llm_cache/
│   └── claude-sonnet-4-6_cache.sqlite          ← Claude API 回應 cache
└── claude-sonnet-4-6_Transformers_.../
    ├── graph.pickle                             ← 知識圖譜
    ├── chunk_embeddings/vdb_chunk.parquet       ← chunk 向量
    ├── entity_embeddings/vdb_entity.parquet     ← entity 向量
    └── fact_embeddings/vdb_fact.parquet         ← fact 向量
```

只有加入**新文件**時，才會對新的部分呼叫 Claude API。

---

## 圖譜節點說明

| 節點類型 | 顏色 | hash_id 前綴 | 內容 |
|---------|------|-------------|------|
| Entity node | 藍色（小） | `entity-` | 實體名稱，如 "cinderella", "montebello" |
| Passage node | 橘色（大） | `chunk-` | 原始文件段落，如 "Montebello is a part of Rockland County." |

---

## 可調整的主要參數

```python
HippoRAG(
    save_dir="outputs/claude",                    # 儲存路徑
    llm_model_name="claude-sonnet-4-6",           # Claude 模型
    embedding_model_name="Transformers/sentence-transformers/all-MiniLM-L6-v2",  # 本地 embedding
    # embedding_model_name="text-embedding-3-small",  # OpenAI embedding（需付費）

    preprocess_chunk_max_token_size=None,         # None = 不拆；設 512 = 每 chunk 最多 512 tokens
    preprocess_chunk_overlap_token_size=128,      # 相鄰 chunk 重疊 token 數
)
```

---

## 可用的 Claude 模型

| 模型 | Model ID | 特性 |
|------|----------|------|
| Claude Opus 4.8 | `claude-opus-4-8` | 最強 |
| Claude Sonnet 4.6 | `claude-sonnet-4-6` | 推薦（速度與品質平衡） |
| Claude Haiku 4.5 | `claude-haiku-4-5-20251001` | 最快、最便宜 |
