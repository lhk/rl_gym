import threading


class Memory:
    def __init__(self):
        # from_state, to_state, action, reward, terminal, length
        # the length is the number of steps between from and to
        # this allows the agents to push observations of arbitrary length
        self.train_queue = [[], [], [], [], [], []]
        self.lock = threading.Lock()

    def __len__(self):
        return len(self.train_queue[0])

    def pop(self, size=1):
        with self.lock:
            retval = [entry[:size] for entry in self.train_queue]
            self.train_queue = [entry[size:] for entry in self.train_queue]

            return retval

    def push(self, from_state, to_state, action, reward, terminal, length):
        with self.lock:
            self.train_queue[0].append(from_state)
            self.train_queue[1].append(to_state)
            self.train_queue[2].append(action)
            self.train_queue[3].append(reward)
            self.train_queue[4].append(terminal)
            self.train_queue[5].append(length)
