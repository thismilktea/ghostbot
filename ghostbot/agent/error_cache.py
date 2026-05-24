import hashlib
import re
import time

ERROR_MEMORY = {}
COOLDOWN_SECONDS = 60

def get_error_fingerprint(error_text):
    clean_text = re.sub(r'\d{4}-\d{2}-\d{2}.*\d{2}:\d{2}:\d{2}', '', error_text)
    return hashlib.md5(clean_text.encode('utf-8')).hexdigest()

def should_diagnose(error_text):
    fingerprint = get_error_fingerprint(error_text)
    now = time.time()
    if fingerprint in ERROR_MEMORY:
        time_since_last = now - ERROR_MEMORY[fingerprint]
        if time_since_last < COOLDOWN_SECONDS:
            print(f"🛡️ [幽灵护盾] 拦截重复报错！剩余冷却: {int(COOLDOWN_SECONDS - time_since_last)}s")
            return False
    ERROR_MEMORY[fingerprint] = now
    return True