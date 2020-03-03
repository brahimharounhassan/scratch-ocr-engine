# from classes.graphPackage import Node
from abc import abstractmethod

class Block:
    
    def __init__(self,label = None):
        self.__label = label

    @property     
    def label(self):
        return self.__label
    
    @label.setter
    def label(self,label):
        self.__label = label 
        
    @abstractmethod
    def get_python_code(self):
        return self.label
    #    return str(self.block_name)

class VerticalBlock(Block):
    
    def __init__(self,block_coordinates,block_type,block_name,block_h): 
        super().__init__(block_name)
        self.__block_type = block_type
        self.__block_coordinates = block_coordinates
        self.__block_h = block_h
        
    
    @property
    def block_type(self):
        return self.__block_type

    @block_type.setter
    def block_type(self,block_type):
        self.__block_type = block_type

    @property
    def block_coordinates(self):
        return self.__block_coordinates

    @block_coordinates.setter
    def set_coordinates(self,block_coordinates):
        self.__block_coordinates = block_coordinates
    
    @property     
    def block_h(self):
        return self.__block_h
    
    @block_h.setter
    def block_h(self,block_h):
        self.__block_h = block_h 
    
    def add_block_h(self,block_h):
        if self.block_h is None:
            self.block_h = BlockH1()
        self.block_h.add_block(block_h)
    def __str__(self):
        if self.block_h is not None:
            return "{}".format(self.block_h)

        
class HorizontalBlock(Block):
    
    def __init__(self,text_content,block_h):
        super().__init__(text_content)
        self.__block_h = block_h
    
    @property
    def block_h(self):
        return self.__block_h
    @block_h.setter
    def block_h(self,block_h):
        self.__block_h = block_h
    
    def add_block(self, block_h):
        self.block_h = block_h
    
    
 
class BlockH1(HorizontalBlock):
    
    def __init__(self,text_content=None,block_h=None):
        super().__init__(text_content,block_h)
    
    def add_block(self,block_h):
        super().add_block(block_h)  

    def __str__(self):
        if super().label and self.block_h is not None:
            return "[Name: {} {}]".format(super().label,self.block_h.__str__())
        elif super().label and self.block_h is None :
            return "[{}] ".format(super().label)
        elif not super().label and self.block_h is not None :
            return "[{}]".format(self.block_h)
        else :
            return "[{}]".format(super().label)
           
class BlockH2(BlockH1):
    
    def __init__(self,text_content = None,block_h1 = None,block_h2 = None):
        super().__init__(text_content,block_h1)
        self.__block_h2 = block_h2
    
    @property
    def block_h2(self):
        return self.__block_h2
    @block_h2.setter
    def block_h2(self,block_h2):
        self.__block_h2 = block_h2
        
    def add_blocks(self, block_h1,block_h2):
        self.block_h1 = block_h1
        self.block_h2 = block_h2

    def __str__(self):

        if super().block_h == self.block_h2:
            return "{}".format(super.block_h().__str__())
        elif super().block_h is not None and self.block_h2 is not None:
            return "[Name: {} - {}]".format(super().label, super().block_h.__str__(),self.block_h2.__str__())
        elif super().block_h is not None and self.block_h2 is None:
            return "{} - {}".format(super().label,super().block_h.__str__())
        elif super().block_h is None and self.block_h2 is not None:
            return "{}".format(self.block_h2.__str__())


                   
class BlockH3(BlockH1):
    
    def __init__(self,text_content=None,block_h1=None,block_h2=None,block_h3=None):
        super().__init__(text_content,block_h1)
        self.block_h2 = block_h2
        self.block_h3 = block_h3
    
    def add_blocks(self, block_h1,block_h2,block_h3):
        self.block_h1 = block_h1
        self.block_h2 = block_h2
        self.block_h3 = block_h3
        
    # def __str__(self):
    #     if self.text_content :
    #         return "{}".format(self.text_content)
    #     elif self.text_content and self.block_h1 is not None :
    #         return "{} content {} ".format(self.text_content,self.block_h1)
    #     elif self.text_content and self.block_h2 is not None:
    #         return "{} content {} ".format(self.text_content,self.block_h2)
    #     else :
    #         return "{} and {} ".format(self.block_h1,self.block_h2)      

