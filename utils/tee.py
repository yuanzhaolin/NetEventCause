#!/usr/bin/python
# -*- coding:utf8 -*-
class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # 刷新输出流，确保内容立即写入

    def flush(self):
        for f in self.files:
            f.flush()