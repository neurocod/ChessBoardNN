import numpy as np

# This network contains variable link count for each neuron. It may contain loops.
# It may contain neurons with permanent activation, preserved between cycles.
# Neurons with permanent activation are contained at indexes [0:self._permanentCount]
# and are processed first, then other neurons are processed until no more new active neurons found
class SemanticNN:
    Activation = np.int8 # use float if need fractions; ints are easier to debug
    #Activation = np.float16
    LinkIndex = np.uint16 # icrease if you have more than 65535 links from one neuron
    def __init__(self, _numNeurons):
        self._numNeurons = _numNeurons

        self._activations = np.zeros(_numNeurons, dtype=self.Activation)
        self._thresholds = np.zeros(_numNeurons, dtype=self.Activation)
        self._sent = np.zeros(_numNeurons, dtype=np.bool_)  # works during current cycle
        self._linkCountByNeuron = np.zeros(_numNeurons, dtype=self.LinkIndex) # by each neuron
        self._permanentCount = 0 # neurons with permanent activation, preserved between cycles
        self._lastPermanentActivations = np.zeros(1, dtype=self.Activation) # to detect changes between cycles
        self._totalLinks = 0
        self._startingLinkByNeuron = []
        self.computeStartingLinks() # call again after NN is filled wiht links

        self._connectionsTo = []
        self._connectionsBias = []

    def computeStartingLinks(self):
        self._startingLinkByNeuron = np.zeros(len(self._linkCountByNeuron), dtype=self.LinkIndex)
        self._totalLinks = 0
        for neuron, links in enumerate(self._linkCountByNeuron):
            self._startingLinkByNeuron[neuron] = self._totalLinks
            self._totalLinks += links

    def forwardStep(self):
        if self._activations[self._permanentCount] >= self._thresholds[self._permanentCount]:
            self.resetNetwork() # see resetNetworkBySpecialNeuron
        else:
            self._sent.fill(False)

        for id in range(self._permanentCount):
            if self._activations[id] < self._thresholds[id]:
                self._activations[id] = 0
        self._activations[self._permanentCount:] = 0

        permanentActivations = self._activations[0:self._permanentCount].copy()
        if np.array_equal(self._lastPermanentActivations, permanentActivations):
            self.loadSensors()
        else:
            self._lastPermanentActivations = permanentActivations

        anySent = False
        totalSent = 0
        # first cycle includes permanent clusters and no check for already _sent
        for id in range(self._numNeurons):
            if self._activations[id] < self._thresholds[id]:
                continue
            anySent = True
            self._sent[id] = True
            totalSent += 1
            startingLink = self._startingLinkByNeuron[id]
            _linkCountByNeuron = self._linkCountByNeuron[id]
            for iLink in range(startingLink, startingLink + _linkCountByNeuron):
                id2 = self._connectionsTo[iLink]
                _connectionsBias = self._connectionsBias[iLink]
                self._activations[id2] += _connectionsBias

        # further cycles - only remaining until processed, and
        while (anySent):
            anySent = False
            for id in range(self._permanentCount, self._numNeurons):
                if self._sent[id] or (self._activations[id] < self._thresholds[id]):
                    continue
                anySent = True
                self._sent[id] = True
                totalSent += 1
                startingLink = self._startingLinkByNeuron[id]
                _linkCountByNeuron = self._linkCountByNeuron[id]
                for iLink in range(startingLink, startingLink + _linkCountByNeuron):
                    id2 = self._connectionsTo[iLink]
                    _connectionsBias = self._connectionsBias[iLink]
                    self._activations[id2] += _connectionsBias

        self.pushEffectors()
        return totalSent
    
    def resetNetworkBySpecialNeuron(self):
        # may require few cycles to take effect. This function is more like documentation,
        # in reality you can connect to this neuron from your network or other neuron
        self._activations[self._permanentCount] = 1

    def resetNetwork(self):
        # override this fn if you want to preserve some neurons
        self._activations.fill(0)
        self._sent.fill(False)
        self._lastPermanentActivations = np.zeros(1, dtype=self.Activation)

    def loadSensors(self):
        pass # override to load sensors

    def pushEffectors(self):
        pass # override to push effectors

    def embedding(self): # override to return more meaningful state
        return self._activations[0:self._permanentCount]

    def getActiveNeurons(self):
        """Get indices of neurons where activation >= threshold"""
        return np.where(self._activations >= self._thresholds)[0].tolist()

if __name__ == "__main__":
    print("Subclass this network and fill with data, look at examples")
    quit(0)