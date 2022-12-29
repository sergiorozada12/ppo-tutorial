class Buffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.states_next = []
        self.logprobs = []
        self.rewards = []
        self.terminals = []
    
    def clear(self):
        self.actions = []
        self.states = []
        self.states_next = []
        self.logprobs = []
        self.rewards = []
        self.terminals = []
        
    def __len__(self):
        return len(self.states)
