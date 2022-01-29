# luseepy package

To set up the PYTHONPATH environment variable:

```bash
source setup.sh
```

Use ```../tests/LunarCalendarTest.py``` to test the calendar.

# Cache

The LunarCalendar class use optional caching utilizing the Python ```shelve```
package. The name of the intended cache can be specified as an argument
of the constructor. The actual file name in which the cache is persisted
will have the ```.db``` extension.

The cache can be optionally deleted once the LunarCalendar object
is destroyed. This behavior is also defined by the constructor based
on the optional argument.
