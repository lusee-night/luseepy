#
# Global cache object
#

# ============= DEPRECATED ======================
# ============= DO NOT USE ======================


import shelve
import atexit

db = shelve.open(".lusee_cache", writeback=True)


def close_db():
    db.close()


atexit.register(close_db)
