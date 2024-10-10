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
        all_entries = []

        # Collect all ontology entries and assign 'Category'
        for category, entries in ontology_data.items():
            for entry in entries:
                entry['Category'] = category  # Use provided categories
                all_entries.append(entry)

        # Add all entries as nodes
        for entry in all_entries:
            eid = entry['id']
            self.graph.add_node(eid, **entry)
            # Extract dependencies from 'Type', 'Proposition', 'Definition', etc.
            dependencies = []
            for field in ['Type', 'Proposition', 'Definition']:
                if field in entry:
                    dependencies.extend(self.extract_dependencies(entry[field]))
            # Add edges for dependencies
            for dep in dependencies:
                if dep != eid and dep in self.graph.nodes:
                    self.graph.add_edge(eid, dep, relation='depends_on')

            # Handle special cases like 'Constructors'
            if 'Constructors' in entry:
                for constructor in entry['Constructors']:
                    for cname, cdef in constructor.items():
                        cid = f"{eid}_{cname}"
                        self.graph.add_node(cid, Category='Constructor', definition=cdef)
                        self.graph.add_edge(eid, cid, relation='has_constructor')

            # Handle 'Contains' relationships
            if 'Contains' in entry:
                for contained in entry['Contains']:
                    self.graph.add_edge(eid, contained, relation='contains')
                    
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
