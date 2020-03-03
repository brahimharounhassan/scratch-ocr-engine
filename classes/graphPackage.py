from collections import defaultdict
from classes.blockPackage import *
# import Graphviz
# import Graphviz
from graphviz import Digraph    
import time

class Node:
    
    init_id = 0
    
    def __init__(self,node_label = None):
        if node_label is None and Node.init_id == 0:
            self.__node_label ='root'
            self.__node_id = Node.init_id
        else:
            self.__node_id = Node.init_id
            self.__node_label = node_label        
        Node.init_id += 1
    
    
    @property
    def node_id(self):
        return self.__node_id

    @node_id.setter
    def node_id(self,node_id):
        self.__node_id = node_id
        
    @property
    def node_label(self):
        return self.__node_label

    @node_label.setter
    def node_label(self,node_label):
        self.__node_label = node_label


class Graph:
    
    visited_node_list=list()
    exclusive_list = list()
    python_dot = Digraph(comment="Python Graph")
    
    def __init__(self,root = None):
        self.__root = root
        if root is not None:
            self.__block_nodes_list = [root]
        else:
            self.__block_nodes_list = list() 
        self.__block_nodes_adjacency_dic = defaultdict(dict)
        self.__output = ""
        Graph.python_dot.node(str(root.node_id),label=r"\\")

    @property
    def root(self):
        return self.__root
    @root.setter
    def root(self,root): 
        self.__root = root
        
    @property
    def block_nodes_list(self):
        return self.__block_nodes_list
    @block_nodes_list.setter
    def block_nodes_list(self,block_nodes_list): 
        self.__block_nodes_list = block_nodes_list
        
    @property
    def block_nodes_adjacency_dic(self):
        return self.__block_nodes_adjacency_dic
    @block_nodes_adjacency_dic.setter
    def block_nodes_adjacency_dic(self,block_nodes_adjacency_dic):
        self.__block_nodes_adjacency_dic = block_nodes_adjacency_dic
              
    @property
    def output(self):
        return self.__output
    @output.setter
    def output(self,output):
        self.__output = output

    def add_node(self,node):
        chaine = "node"
        self.block_nodes_list.append(node)
        Graph.python_dot.node(str(node.node_id), node.node_label)
        if node.block_h is not None:
            chaine+=str(node.node_id)
            with Graph.python_dot.subgraph(name='cluster'+chaine) as p:
                p.attr(color="blue")
                p.node_attr.update(style='filled',color="pink")
                p.node(chaine,node.block_h.__str__(),shape="box")
                p.edge(str(node.node_id),chaine)
                p.attr(label='')

        
    def add_edge(self,edge):
        if edge.block_node1 in self.block_nodes_adjacency_dic.keys():
            self.block_nodes_adjacency_dic[edge.block_node1].append(edge)
        else:
            self.block_nodes_adjacency_dic[edge.block_node1]= list()
            self.block_nodes_adjacency_dic[edge.block_node1].append(edge)
        
    def display_graph(self):
        for node in self.block_nodes_adjacency_dic.values():
            for block in node:
                print(block)
      
    def show_dfs(self):
        self.dfs(self.root,Graph.visited_node_list)
        
    def dfs(self,node,visited_node_list):
        if node not in visited_node_list:
            # faire appel a la methode getPython code de node
            print(node)

            if node in self.block_nodes_adjacency_dic:
                for edge in self.block_nodes_adjacency_dic[node]:
                    visited_node_list.append(node)
                    self.dfs(edge.block_node2,Graph.visited_node_list)
                    
    def build_graphviz_file(self):
        for node in self.block_nodes_adjacency_dic.values():
            for block in node:
                Graph.python_dot.edge(str(block.block_node1.node_id),str(block.block_node2.node_id))
        Graph.python_dot.attr(overlap="false")
        # print(Graph.python_dot.source)
        Graph.python_dot.render('test-output/scratch_graph.gv', view=True)
        'test-output/scratch_graph.gv.pdf'
