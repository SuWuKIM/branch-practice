# scripts/net_check.py
import sys, requests
url = sys.argv[1] if len(sys.argv) > 1 else "https://openai.com/news/rss.xml"
r = requests.get(url, timeout=15)
print("status_code =", r.status_code, "len =", len(r.text))
print("sample:", r.text[:200].replace("\n", " "))
