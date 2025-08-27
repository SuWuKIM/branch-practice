# scripts/db_check.py
import os, sqlite3, sys

path = os.path.join('data', 'processed', 'app.db')
print('DB exists?', os.path.exists(path))
if not os.path.exists(path):
    sys.exit(0)

conn = sqlite3.connect(path)
cur = conn.cursor()

# documents 테이블 존재 여부
cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='documents'")
exists = cur.fetchone() is not None
print('documents table exists?', exists)

if exists:
    cur.execute("SELECT COUNT(*) FROM documents")
    print('documents count =', cur.fetchone()[0])
    # 샘플 3개 확인(있다면)
    cur.execute("SELECT title, url FROM documents LIMIT 3")
    rows = cur.fetchall()
    if rows:
        print('sample rows:', rows)

conn.close()
