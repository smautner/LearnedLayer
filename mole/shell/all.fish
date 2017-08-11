echo (seq 1 54 )|string split " " | parallel python runner.py
