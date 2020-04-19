from .walker import RandomWalker


class Node2Path:
    def __init__(self, graph, walk_length, num_walks, p=1.0, q=1.0, workers=1):

        self.graph = graph
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.workers = workers

    def get_path(self):

        self.walker = RandomWalker(self.graph, p=self.p, q=self.q)

        print("Preprocess transition probs...")
        self.walker.preprocess_transition_probs()

        self.sentences = self.walker.simulate_walks(
            num_walks=self.num_walks,
            walk_length=self.walk_length,
            workers=self.workers,
            verbose=1,
        )
        return self.sentences



