# !/usr/bin/python
# -*-coding:utf8
from classes.scratchTree import BlockNode

class When(BlockNode):
    def __init__(self,block_coordinates,block_type,block_name,block_h=None):
        super().__init__(block_coordinates,block_type,block_name,block_h)
        
    def get_python_code(self):
        return "start robot"
    
class Wait(BlockNode):
    def __init__(self,block_coordinates,block_type,block_name,block_h=None):
        super().__init__(block_coordinates,block_type,block_name,block_h)

    def get_python_code(self):
        return "time.sleep(2)"

class Repeat(BlockNode):
    def __init__(self,block_coordinates,block_type,block_name,block_h=None):
        super().__init__(block_coordinates,block_type,block_name,block_h)
    def get_python_code(self):
            return "for i in range(0,10):"

class Forever(BlockNode):
    def __init__(self,block_coordinates,block_type,block_name,block_h=None):
        super().__init__(block_coordinates,block_type,block_name,block_h)
    def get_python_code(self):
            return "while True:"

class If(BlockNode):
    def __init__(self,block_coordinates,block_type,block_name,block_h=None):
        super().__init__(block_coordinates,block_type,block_name,block_h)
    def get_python_code(self):
            return "if :"

class IfElse(BlockNode):
    def __init__(self,block_coordinates,block_type,block_name,block_h=None):
        super().__init__(block_coordinates,block_type,block_name)
    def get_python_code(self):
            return "if else:"
class WaitUntil(BlockNode):
    def __init__(self,block_coordinates,block_type,block_name,block_h=None):
        super().__init__(block_coordinates,block_type,block_name,block_h)
    def get_python_code(self):
            return "time.sleep((condition))"
class RepeatUntil(BlockNode):
    def __init__(self,block_coordinates,block_type,block_name,block_h=None):
        super().__init__(block_coordinates,block_type,block_name,block_h)
    def get_python_code(self):
            return "While (condition):"

class Stop(BlockNode):
    def __init__(self,block_coordinates,block_type,block_name,block_h=None):
        super().__init__(block_coordinates,block_type,block_name,block_h)
    def get_python_code(self):
            return "exit()"

class Move(BlockNode):
    def __init__(self,block_coordinates,block_type,block_name,block_h=None):
        super().__init__(block_coordinates,block_type, block_name,block_h)
        
    def get_python_code(self):
        return "move 10 steps"
class Turn(BlockNode):
    def __init__(self,block_coordinates,block_type,block_name,block_h=None):
        super().__init__(block_coordinates,block_type,block_name,block_h)
        
    def get_python_code(self):
        return "turn left"
