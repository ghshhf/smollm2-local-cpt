@echo off
cd /d D:\work\_local_ai\train_15b
venv\Scripts\python.exe -u tools\ai_frontier_crawl.py --days 1 >> logs\frontier_crawl.log 2>&1
