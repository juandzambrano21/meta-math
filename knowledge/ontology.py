# knowledge/ontology.py

import json
import networkx as nx
import re

class Ontology:
    def __init__(self, ontology_json):
        """
        Initializes the Ontology from a JSON string.
        """
        self.graph = nx.DiGraph()
        self.current_node_id = 'Type'  # Starting node in the ontology
        self.load_ontology(ontology_json)

    def load_ontology(self, ontology_json):
        """
        Loads the ontology from a JSON string.
        """
        ontology_data = json.loads(ontology_json)
        self.parse_ontology(ontology_data)

    def parse_ontology(self, ontology_data):
        """
        Parses the ontology data and constructs the graph.
        """
        # Add Universes
        for universe in ontology_data.get('Universes', []):
            uid = universe['id']
            self.graph.add_node(uid, kind='Universe', **universe)
            # If 'Contains' is specified, add edges
            for contained in universe.get('Contains', []):
                self.graph.add_edge(uid, contained, relation='contains')

        # Add Types
        for type_def in ontology_data.get('Types', []):
            tid = type_def['id']
            self.graph.add_node(tid, kind='Type', **type_def)
            # Parse 'Type' field to find dependencies
            dependencies = self.extract_dependencies(type_def.get('Type', ''))
            for dep in dependencies:
                if dep != tid and dep in self.graph.nodes:
                    self.graph.add_edge(tid, dep, relation='depends_on')

        # Add Theorems
        for theorem in ontology_data.get('Theorems', []):
            tid = theorem['id']
            self.graph.add_node(tid, kind='Theorem', **theorem)
            # Parse 'Proposition' to find dependencies
            dependencies = self.extract_dependencies(theorem.get('Proposition', ''))
            for dep in dependencies:
                if dep != tid and dep in self.graph.nodes:
                    self.graph.add_edge(tid, dep, relation='uses')

        # Add HITypeDefinitions
        for hi_type in ontology_data.get('HITypeDefinitions', []):
            hid = hi_type['id']
            self.graph.add_node(hid, kind='HIType', **hi_type)
            # Add constructors
            for constructor in hi_type.get('Constructors', []):
                for cname, cdef in constructor.items():
                    cid = f"{hid}_{cname}"
                    self.graph.add_node(cid, kind='Constructor', definition=cdef)
                    self.graph.add_edge(hid, cid, relation='has_constructor')

        # Add MetaMathematics
        for meta in ontology_data.get('MetaMathematics', []):
            mid = meta['id']
            self.graph.add_node(mid, kind='MetaMathematics', **meta)
            # Parse 'Proposition' or 'Definition' to find dependencies
            dependencies = self.extract_dependencies(meta.get('Proposition', '') + meta.get('Definition', ''))
            for dep in dependencies:
                if dep != mid and dep in self.graph.nodes:
                    self.graph.add_edge(mid, dep, relation='references')

    def extract_dependencies(self, text):
        """
        Extracts dependencies (node IDs) from the given text.
        """
        # Simple parser to extract words starting with uppercase letters
        return re.findall(r'\b[A-Z][A-Za-z0-9_]*\b', text)

    def get_neighbors(self, node_id):
        """
        Returns the neighbors of a given node.
        """
        return list(self.graph.successors(node_id))

    def get_node(self, node_id):
        """
        Returns the node data for a given node_id.
        """
        return self.graph.nodes[node_id]

    def navigate(self, action):
        """
        Navigates the ontology graph based on the action.
        :param action: The action to take, e.g., ('MoveTo', node_id).
        """
        if action[0] == 'MoveTo':
            node_id = action[1]
            if node_id in self.graph.successors(self.current_node_id):
                self.current_node_id = node_id
                print(f"Moved to node {node_id}")
            else:
                print(f"Cannot move to node {node_id} from {self.current_node_id}")
        elif action[0] == 'Inspect':
            node_id = action[1]
            if node_id in self.graph.nodes:
                node_info = self.get_node(node_id)
                print(f"Inspecting node {node_id}: {node_info}")
            else:
                print(f"Node {node_id} does not exist in the ontology")
        else:
            print(f"Unknown action {action}")

    def get_current_node_info(self):
        """
        Returns information about the current node.
        """
        if self.current_node_id in self.graph.nodes:
            return self.graph.nodes[self.current_node_id]
        else:
            return None

    def reset(self):
        """
        Resets the current node to the starting node.
        """
        self.current_node_id = 'Type'
