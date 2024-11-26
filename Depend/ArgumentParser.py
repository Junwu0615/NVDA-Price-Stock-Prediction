# -*- coding: utf-8 -*-
"""
@author: PC
Update Time: 2024-11-24
"""

from argparse import ArgumentParser, Namespace

class AP:
    def __init__(self, obj):
        self.obj = obj

    def parse_args(self) -> Namespace:
        parse = ArgumentParser()
        parse.add_argument("-open", "--open",
                           help="Add open feature ?",
                           default="T", type=str)

        parse.add_argument("-high", "--high",
                           help="Add high feature ?",
                           default="T", type=str)

        parse.add_argument("-low", "--low",
                           help="Add low feature ?",
                           default="T", type=str)

        parse.add_argument("-vol", "--volume",
                           help="Add volume feature ?",
                           default="T", type=str)

        parse.add_argument("-utm", "--use_trained_model",
                           help="Use Trained Model ?",
                           default="F", type=str)

        return parse.parse_args()

    def config_once(self):
        args = self.parse_args()
        self.obj.open = args.open
        self.obj.high = args.high
        self.obj.low = args.low
        self.obj.volume = args.volume
        self.obj.use_trained_model = args.use_trained_model