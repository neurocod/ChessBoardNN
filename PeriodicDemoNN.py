from ConsoleSemanticNN import ConsoleSemanticNN
import numpy as np
import time

class PeriodicDemoNN(ConsoleSemanticNN):
    """
    Outputs symbols A B A B ... forever
    """
    def __init__(self):
        super().__init__(7)
        self._permanentCount = 1
        self._thresholds = np.array([1,1,0,2,1,1,1], dtype=self.Activation)
        self._linkCountByNeuron = np.array([2,0,2,2,2,0,0], dtype=self.LinkIndex)
        self._connectionsTo = np.array([3,4,4,3,0,6,0,5], dtype=self.LinkIndex)
        self._connectionsBias = np.array([1,-1,1,1,-1,1,1,1], dtype=self.Activation)
        self._sensors = {'reset':1}
        self._effectors = {6:'b',5:'a'}
        self.computeStartingLinks()

if __name__ == "__main__":
    print("This network writes periodic sequence - it uses long short-term memory to store it's state and implements oscillator. " +
          "It changes it's state only when input is triggered. It is useful for chessboard NN to change state from White to Black moves, etc. " +
          "In this case, input is triggered at each cycle.\nOutput:")
    demoNet = PeriodicDemoNN()

    for i in range(50):
        demoNet.forwardStep()

    print("... (forever)\nWith activations at the end of each cycle:")

    for i in range(10):
        demoNet.forwardStep()
        print(demoNet._activations)

    print("""Neurons with descriptions:
[n].threshold comment
[0].1 LSTM neuron: is it B?
[1].1 reset network (not used here)
[2].0 always active
[3].2 gate for B
[4].1 gate for A
[5].1 a (out)
[6].1 b (out)
        """)