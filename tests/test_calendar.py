from lusee.LunarCalendar import LunarCalendar


def test_lunar_day_2503():
    lc = LunarCalendar()
    start, end = lc.get_lunar_start_end(2503)
    start.precision = end.precision = 6
    fmt = "%Y %m %d %H %M %S"
    assert start.strftime(fmt) == "2025 04 27 19 29 24"
    assert end.strftime(fmt) == "2025 05 27 07 05 14"
