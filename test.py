import urllib3
urllib3.disable_warnings()
import requests
import trafilatura

r = requests.get(
    'https://simple.wikipedia.org/wiki/Eiffel_Tower',
    verify=False,
    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
)
print('Status:', r.status_code)

content = trafilatura.extract(r.text, include_tables=True, no_fallback=False)
print('Content length:', len(content) if content else 0)
print('Preview:', content[:300] if content else 'NONE')