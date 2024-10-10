# environments/continuous_navigation.py

import numpy as np

class ContinuousNavigationEnv:
    def __init__(self, area_size=(50.0, 50.0), goal_position=(45.0, 45.0), obstacles=None, motion_noise_cov=None, observation_noise_cov=None, goal_threshold=1.0, ontology=None):
        """
        Initializes the Continuous Navigation Environment.

        :param area_size: Tuple representing the size of the 2D area (width, height).
        :param goal_position: Tuple representing the goal's coordinates.
        :param obstacles: List of dictionaries with 'position' and 'radius'.
        :param motion_noise_cov: Covariance matrix for motion noise.
        :param observation_noise_cov: Covariance matrix for observation noise.
        :param goal_threshold: Distance threshold to consider the goal as reached.
        :param ontology: An instance of Ontology for mapping positions to ontology nodes.
        """
        self.area_size = np.array(area_size)
        self.goal_position = np.array(goal_position)
        self.obstacles = obstacles if obstacles else []
        self.motion_noise_cov = motion_noise_cov if motion_noise_cov is not None else np.eye(2) * 0.1
        self.observation_noise_cov = observation_noise_cov if observation_noise_cov is not None else np.eye(2) * 0.5
        self.goal_threshold = goal_threshold
        self.ontology = ontology  # Assign the ontology to the environment
        self.reset()

    def reset(self):
        """
        Resets the environment to the initial state.

        :return: Initial observation as a numpy array.
        """
        self.state = np.array([5.0, 5.0], dtype=float)
        self.done = False
        return self.get_observation()

    def step(self, action):
        """
        Executes an action in the environment.

        :param action: Tuple representing acceleration vector (ax, ay).
        :return: Tuple (observation, reward, done, info).
        """
        if self.done:
            raise Exception("Environment has been terminated. Please reset.")

        # Apply action with motion noise
        acceleration = np.array(action)
        motion_noise = np.random.multivariate_normal(np.zeros(2), self.motion_noise_cov)
        self.state += acceleration + motion_noise

        # Enforce area boundaries
        self.state = np.clip(self.state, 0.0, self.area_size)

        # Check for collisions with obstacles
        collision = False
        for obs in self.obstacles:
            distance = np.linalg.norm(self.state - obs['position'])
            if distance <= obs['radius']:
                collision = True
                break

        # Map position to ontology node
        ontology_node = self.map_position_to_ontology(self.state)

        # Calculate distance to goal
        distance_to_goal = np.linalg.norm(self.state - self.goal_position)

        # Check if goal is reached
        if distance_to_goal <= self.goal_threshold:
            self.done = True
            reward = 100.0
        elif collision:
            reward = -10.0  # Negative reward for collision
        else:
            reward = -1.0  # Small negative reward for each step taken

        observation = self.get_observation()
        info = {'distance_to_goal': distance_to_goal, 'collision': collision, 'ontology_node': ontology_node}

        return observation, reward, self.done, info

    def get_observation(self):
        """
        Generates an observation with noise.

        :return: Observation as a numpy array.
        """
        observation_noise = np.random.multivariate_normal(np.zeros(2), self.observation_noise_cov)
        observation = self.state + observation_noise
        return observation

    def map_position_to_ontology(self, position):
        """
        Maps the agent's position to an ontology node.

        :param position: The agent's position as a numpy array.
        :return: The ontology node ID, if any.
        """
        if self.ontology:
            # Custom mapping logic between positions and ontology nodes
            x, y = position
            area_width, area_height = self.area_size
            if x < area_width / 2 and y < area_height / 2:
                return 'Type'
            elif x >= area_width / 2 and y < area_height / 2:
                return 'Term'
            elif x < area_width / 2 and y >= area_height / 2:
                return 'Univalence'
            else:
                return 'Circle'
        else:
            return None

    def render(self):
        """
        Renders the current state of the environment.
        """
        grid_size = 20
        grid = [['.' for _ in range(grid_size)] for _ in range(grid_size)]

        # Mark obstacles
        for obs in self.obstacles:
            x, y = obs['position']
            r = obs['radius']
            grid_x = int(x / self.area_size[0] * (grid_size - 1))
            grid_y = int(y / self.area_size[1] * (grid_size - 1))
            radius = int(r / min(self.area_size) * (grid_size - 1))
            for i in range(max(0, grid_y - radius), min(grid_size, grid_y + radius + 1)):
                for j in range(max(0, grid_x - radius), min(grid_size, grid_x + radius + 1)):
                    if np.linalg.norm(np.array([j, i]) - np.array([grid_x, grid_y])) <= radius:
                        grid[i][j] = 'X'

        # Mark goal
        goal_x, goal_y = self.goal_position
        grid_goal_x = int(goal_x / self.area_size[0] * (grid_size - 1))
        grid_goal_y = int(goal_y / self.area_size[1] * (grid_size - 1))
        grid[grid_goal_y][grid_goal_x] = 'G'

        # Mark agent
        agent_x, agent_y = self.state
        grid_agent_x = int(agent_x / self.area_size[0] * (grid_size - 1))
        grid_agent_y = int(agent_y / self.area_size[1] * (grid_size - 1))
        grid[grid_agent_y][grid_agent_x] = 'A'

        # Print grid
        print('\n'.join([' '.join(row) for row in grid]))
        print()
