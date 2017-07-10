#!/usr/bin/env python
# encoding: utf-8

import multiprocessing

def f(x): return x*x

pool = multiprocessing.Pool(processes=4)

print(pool.map(f, range(5)))
