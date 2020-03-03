# !/usr/bin/python
# -*-coding:utf8
from classes.graph1 import BlockNode

class Move(BlockNode):
    def __init__(self,block_coordinates,block_type,name):
        super().__init__(block_coordinates,block_type, name)
        
    def get_python_code(self):
        return "move 10 steps"
class Turn(BlockNode):
    def __init__(self,block_coordinates,block_type,name):
        super().__init__(block_coordinates,block_type,name)
        
    def get_python_code(self):
        return "turn left"

def turnRight(*args):
    """
    turn right the robot from degres value.
    """
    return f"print('Turn Right')"

def turnLeft(angle = 15):
    """
    turn left the robot from degres value.
    """
    print("Turn Left")

def goTo(x = 0, y = 0):
    """
    move the robot to coordinates given by x and y.
    """
    print("Go To")
    