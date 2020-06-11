from abc import abstractmethod
from classes.graphPackage import Node
from classes.blockPackage import VerticalBlock,BlockH1

  
class BlockNode(VerticalBlock,Node):
    
    def __init__(self, block_coordinates,block_type,block_name,block_h):
        super().__init__(block_coordinates,block_type,block_name,block_h)
        Node.__init__(self,block_name)

        
    @abstractmethod
    def get_python_code(self):
       return str(super().node_label)
   
    def get_graphic_rep(self):
       return str(super().node_label)


    def  __str__(self):
        if super().block_h is not None:
            return f"Name: {self.node_label} - Id : {self.node_id} -[ {super().block_h.__str__()}]"
        else:
            return f"Name: {self.node_label} - Id : {self.node_id}"
  
  
class BlockEdge:
    
    def __init__(self,block_node1,block_node2):
        self.__block_node1 = block_node1
        self.__block_node2 = block_node2
        
    @property
    def block_node1(self):
        return self.__block_node1
        
    @block_node1.setter
    def block_node1(self,block_node1):
        self.__block_node1 = block_node1
        
    @property
    def block_node2(self):
        return self.__block_node2
        
    @block_node2.setter
    def block_node2(self,block_node2):
        self.__block_node2 = block_node2

    def __eq__(self,block_edge):
        if not isinstance(block_edge,BlockEdge):
            return False
        return (self.block_node1.block_id == block_edge.block_node1.block_id and self.block_node2.block_id ==  block_edge.block_node2.block_id) or self.block_node1.block_id == block_edge.block_node2.block_id and self.block_node2.block_id ==  block_edge.block_node1.block_id
       
    def __str__(self):
        return f"[{self.block_node1.__str__()}] - [{self.block_node2.__str__()}]"
