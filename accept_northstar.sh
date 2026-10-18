#!/bin/bash
# Acceptance via Ollama's own HTTP API (no custom inference code).
export PATH="/c/Users/admin/AppData/Local/Programs/Ollama:$PATH"
i=0
while IFS= read -r q; do
  i=$((i+1))
  printf "\n=== Q%d: %s\n" "$i" "$q"
  curl -s --max-time 90 http://127.0.0.1:11434/api/chat \
    -d "$(python -c "
import json,sys
print(json.dumps({'model':'northstar','messages':[{'role':'user','content':sys.argv[1]}],'stream':False,'options':{'temperature':0,'num_predict':120}}))
" "$q")" | python -c "
import json,sys
d=json.load(sys.stdin)
print('A:', d.get('message',{}).get('content','').replace(chr(10),' ')[:300])
"
done <<'EOF'
你是谁
你叫什么名字
你是人吗
你是中文助手吗
你是ChatGPT吗
你住在哪里
主人今天有点累
讲个笑话吧
你都会干什么
晚安
EOF
echoACC_DONE