#!/bin/bash
ECS="http://47.99.131.12:8081"
PROBE_CMD='echo ALIVE_PROBE_$(date +%s) > /tmp/pn1.txt; curl -s -m60 -X POST "http://47.99.131.12:8081/upload?name=probe_node1_v2.txt&key=k0u4mail" --data-binary @/tmp/pn1.txt; echo done'
echo "=== POST probe task ==="
curl -s -m20 -X POST "$ECS/task" -H "Content-Type: application/json" -d "{\"cmd\":\"$PROBE_CMD\"}"
echo
echo "=== immediate re-check /task (empty => consumed) ==="
curl -s -m20 "$ECS/task"
echo
echo "=== check probe result file on ECS ==="
curl -s -m20 "$ECS/file/probe_node1_v2.txt"
echo
