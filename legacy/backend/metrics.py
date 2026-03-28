from prometheus_client import Counter, Histogram, Gauge

DETECTIONS_TOTAL = Counter(
    "kavach_detections_total",
    "Total number of detection requests processed",
    ["source", "tier"]
)

MODEL_LATENCY = Histogram(
    "kavach_model_latency_seconds",
    "Latency of individual model inference",
    ["model_name"]
)

CACHE_HITS = Counter(
    "kavach_cache_hits_total",
    "Total number of cache hits for detection requests"
)

CONFIDENCE_HISTOGRAM = Histogram(
    "kavach_confidence_score",
    "Distribution of confidence scores",
    ["verdict"]
)

ACTIVE_SCANS = Gauge(
    "kavach_active_scans",
    "Number of currently active scan requests"
)
