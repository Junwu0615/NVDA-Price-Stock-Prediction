# -*- coding: utf-8 -*-
"""
@author: PC
Update Time: 2024-11-24
"""

from Depend.Model import Model
from Depend.ArgumentParser import AP

class Entry:
    def __init__(self):
        self.open = None
        self.high = None
        self.low = None
        self.volume = None
        self.use_trained_model = None

    def main(self):
        ap = AP(self)
        ap.config_once()
        model = Model(self)
        model.main()

if __name__ == '__main__':
    entry = Entry()
    entry.main()