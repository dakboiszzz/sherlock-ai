import json
import os
data = []
with open('sherlock.json', 'r', encoding='utf-8') as f:
    file_content = f.read().strip()
    data = json.loads(file_content)
#print(json.dumps(data[0],indent = 4))

