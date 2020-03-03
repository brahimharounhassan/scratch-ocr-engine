# !/usr/bin/python
# -*-coding:utf8
from classes.graph1 import BlockNode

class When(BlockNode):
    def __init__(self,block_coordinates,block_type,name):
        super().__init__(block_coordinates,block_type,name)
        
    def get_python_code(self):
        return "start robot"