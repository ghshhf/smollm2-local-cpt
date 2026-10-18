cd /mnt/workspace/node1
curl -s -m 60 -o refine_reserve.py "http://47.99.131.12:8081/file/refine_reserve.py"
if [ ! -s refine_reserve.py ]; then echo "DOWNLOAD_FAIL"; exit 1; fi
python3 -c "import py_compile;py_compile.compile('refine_reserve.py',doraise=True);print('SYNTAX_OK')"

cat > /tmp/run_refine.sh <<'EOF'
cd /mnt/workspace/node1
TOKEN_BUDGET=24600000 python3 -u refine_reserve.py > /tmp/refine1.log 2>&1
cp /tmp/refine1.log /mnt/workspace/node1/refine_run.log
curl -s -m 120 -F "file=@/tmp/refine1.log" "http://47.99.131.12:8081/upload?name=refine_run_a1.txt&key=k0u4mail"
EOF
chmod +x /tmp/run_refine.sh
( nohup bash /tmp/run_refine.sh >/dev/null 2>&1 & )
sleep 2
echo "LAUNCHED pid=$(pgrep -f 'refine_[r]eserve.py' | tr '\n' ' ')"
