import sqlite3, os

db = 'outputs/2wikimultihopqa/llm_cache/claude-sonnet-4-6_cache.sqlite'
total_docs = 6119

if os.path.exists(db):
    conn = sqlite3.connect(db)
    calls = conn.execute('SELECT COUNT(*) FROM cache').fetchone()[0]
    conn.close()
    docs = calls // 2
    pct = docs / total_docs * 100
    remaining = total_docs - docs
    print(f'OpenIE 進度: {docs}/{total_docs} ({pct:.1f}%)')
    print(f'已完成 API 呼叫: {calls}')
    print(f'剩餘文件: {remaining} 篇')
else:
    print('尚未開始（找不到 cache 檔案）')
