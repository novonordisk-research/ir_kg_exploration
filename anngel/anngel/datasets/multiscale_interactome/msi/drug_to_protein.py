from .node_to_node import NodeToNode

class DrugToProtein(NodeToNode):
	def __init__(self, directed, file_path, sep = "\t"):
		super().__init__(directed, file_path, sep)