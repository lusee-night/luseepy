#! /usr/bin/env python

import argparse
import time
from lusee.LunarCalendar import LunarCalendar

#######################################
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbose",  action='store_true', help="Verbose mode")
parser.add_argument("-c", "--cleanup",  action='store_true', help="Remove cache file on exit")
parser.add_argument("-f", "--cachefile",type=str,            help="The optional cache file", default='')
#######################################

args    = parser.parse_args()

cachefile   = args.cachefile
cleanup     = args.cleanup
verb        = args.verbose
#######################################

lc = LunarCalendar(cachefile, cleanup, verb)
timer_start = time.time()
(start, end) = lc.get_lunar_start_end(2503)
timer_end = time.time()

if verb:
    print("Lunar start/end calculation time: %5.2f seconds" % (timer_end - timer_start))

start.precision, end.precision = (6, 6)

fmt = "%Y %m %d %H %M %S"
starts = start.strftime(fmt)
ends = end.strftime(fmt) 
print(starts)
print(ends)
assert(starts == "2025 04 27 19 29 24")
assert(ends == "2025 05 27 07 05 14")
print ("Asserts passed.")
