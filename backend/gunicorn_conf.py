
import multiprocessing
import os

# Gunicorn Configuration for KAVACH-AI

# Bind to all interfaces
bind = "0.0.0.0:8000"

# Worker Setup
# Rule of thumb: 2-4 x $(NUM_CORES)
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "uvicorn.workers.UvicornWorker"

# Timeouts
# AI processing can be slow, increase timeout
timeout = 120
keepalive = 5

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Process Naming
proc_name = "kavach_api"
