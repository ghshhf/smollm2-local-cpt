cd /mnt/workspace/node1
REF_DIR=/mnt/workspace/node1/chain/corpus_reserve_refined python3 -u verify_refine.py > /tmp/vf3.txt 2>&1
python3 -u diag_div.py > /tmp/dd3.txt 2>&1
cat /tmp/vf3.txt /tmp/dd3.txt > /tmp/all3.txt
curl -s -m 60 --data-binary "@/tmp/all3.txt" "http://47.99.131.12:8081/upload?name=verify_out_v3.txt&key=k0u4mail" >/dev/null 2>&1
echo ok
