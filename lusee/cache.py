#
# Global cache object
#
import shelve

db = shelve.open(".lusee_cache", writeback=True)
