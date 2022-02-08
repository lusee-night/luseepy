import sys

sys.path.append('.')

import lusee

(start, end) = lusee.calendar.get_lunar_start_end(2503)

start.precision, end.precision = (6, 6)

fmt = "%Y %m %d %H %M %S %f"
print(start.strftime(fmt))
print(end.strftime(fmt))
