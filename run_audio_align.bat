@echo off
D:/work/_local_ai/train_15b/venv/Scripts/python.exe D:/work/_local_ai/train_15b/tools/audio_attach.py --llm D:/work/_local_ai/train_15b/backups/0.3B/polaris-03b_20261018 --stage align --steps 2000 --use-cache --val-speakers S0007 --save D:/work/_local_ai/train_15b/models/audio_03b_align >> D:/work/_local_ai/train_15b/logs/audio_align.log 2>&1
