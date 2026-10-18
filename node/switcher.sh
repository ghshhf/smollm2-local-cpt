#!/bin/sh
# switcher.sh — 等 0.3B 三档归档完成后, 自动把 v5 阶梯换成 v6(04B 用准备好的全部资料)
for i in $(seq 1 300); do
  grep -q "\[R10\] 0.3B 归档" /root/autopilot5.log 2>/dev/null && break
  sleep 10
done
sleep 5
A=gpu_; B=utopilot5.py
for p in $(ps -eo pid,args | grep -E "$A$B" | grep -v grep | awk '{print $1}'); do kill -9 $p 2>/dev/null; done
E=expand_to_; F=target.py
for p in $(ps -eo pid,args | grep -E "$E$F" | grep -v grep | awk '{print $1}'); do kill -9 $p 2>/dev/null; done
sleep 2
rm -f /tmp/autopilot5.lock
curl -s -o /root/gpu_autopilot6.py "http://47.99.131.12:8081/file/gpu_autopilot6.py?key=k0u4mail"
cd /root
setsid nohup python3 -u /root/gpu_autopilot6.py > /root/autopilot6.log 2>&1 < /dev/null &
sleep 20
echo "SWITCHED at $(date +%H:%M:%S)" > /root/switcher_status.txt
tail -6 /root/autopilot6.log >> /root/switcher_status.txt
