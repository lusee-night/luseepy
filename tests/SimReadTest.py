#!/usr/bin/env python

import lusee
import sys

if (len(sys.argv)<=1):
    print ("Specify filename on command line.")
fname = sys.argv[1]
print (f"Attempting to read {fname}...")
d = lusee.Data(sys.argv[1])
subdata =-d[:,"12C",:]
print ("OK.")

