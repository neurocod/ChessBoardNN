try:
    from .SemanticNN import SemanticNN
except ImportError:
    from SemanticNN import SemanticNN
import numpy as np
import re

# Extends SemanticNN to read input from console, write output to console
# It contains sensors (string->neuron) and effectors (neuron->string)
# Input is tokenized by words and punctuation, case sensitive
class ConsoleSemanticNN(SemanticNN):
    def __init__(self, numNeurons):
        super().__init__(numNeurons)
        self._awaitingInput = []
        self._sensors = {}  # token to neuron
        self._effectors = {}  # neuron to token
        self._printToConsole = True # False -> accumulate to _outputBuffer only
        self._outputBuffer = [] # for Unit Tests etc

    def loadSensors(self): # Read input from console, tokenize it, and activate corresponding neurons
        if len(self._awaitingInput) == 0:
            return
        activated = self.activateByTokens(self._awaitingInput[0:1])
        self._awaitingInput.pop(0)
        # print(f"Tokens: {tokens}\nNeurons activated by tokens: {activated}")

    def pushEffectors(self): # neurons -> console output
        for neuron_index, text in self._effectors.items():
            if self._activations[neuron_index] >= self._thresholds[neuron_index]:
                self._outputBuffer.append(text)
        if self._outputBuffer and self._printToConsole:
            print(self.flushOutput(), end="")

    def flushOutput(self)->str:
        ret = ''.join(self._outputBuffer)
        self._outputBuffer = []
        return ret

    def tokenize(self, text):
        """
        Tokenize input text into words and punctuation
        Return list of tokens
        Split by whitespace and punctuation, keep punctuation
        """
        tokens = re.findall(r'\w+|[^\w\s]', text.lower())
        return tokens

    def activateByTokens(self, tokens): # activate corresponding neurons and return their list
        activated_neurons = []
        for token in tokens:
            if token in self._sensors:
                neuron_id = self._sensors[token]
                self._activations[neuron_id] = self._thresholds[neuron_id]
                activated_neurons.append(neuron_id)

        return activated_neurons

    def printActivations(self, array, printColumnEvery=20, neuronsPerRow = 40):
        for i in range(0, len(array), neuronsPerRow):
            chunk = array[i:i+neuronsPerRow]
            # Convert numbers to strings, replace 0 with '.'
            if i==0:
                formatted_chunk = ["["]
            else:
                formatted_chunk = ["\n "]
            firstNum = True
            for i, activation in enumerate(chunk):
                if not firstNum and i % printColumnEvery == 0:
                    formatted_chunk.append('|')
                firstNum = False
                if activation == 0:
                    formatted_chunk.append('.')
                else:
                    formatted_chunk.append(str(activation))
            print("".join(formatted_chunk), end="")
        print("]")
    
    def printEmbedding(self):
        self.printActivations(self.embedding())

    def interactiveMode(self): # accept console input and process it
        print("""Semantic Neural Network - Interactive Mode
Type 'quit' to exit, 'info' for network info, 'all' to see all neurons activation, 'perm' to see permanent neurons, 'repr' to see more specific representation/embedding, 'sensors' to see sensors, 'effectors' to see effectors.
        """)
        print("_" * 50)

        while True:
            try:
                if len(self._awaitingInput) == 0:
                    userInput = input("\n> ").strip()
                    if not userInput or userInput == '':
                        continue

                    lower = userInput.lower()
                    if lower == 'quit':
                        break
                    elif lower == 'info':
                        self.printNetworkInfo()
                        continue
                    elif lower == 'repr':
                        self.printEmbedding()
                        continue
                    elif lower == 'perm':
                        self.printActivations(self._activations[0:self._permanentCount])
                        continue
                    elif lower == 'all':
                        self.printActivations(self._activations, 20, 100)
                        continue
                    elif lower == 'sensors':
                        self.printSensors()
                        continue
                    elif lower == 'effectors':
                        self.printEffectors()
                        continue
                    self.processInput(userInput)

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")

    def printNetworkInfo(self):
        print(f"Neurons: {self._numNeurons}")
        print(f"Links: {self._totalLinks}")
        print(f"Active neurons: {int(np.sum(self._activations))}")
        print(f"Average connectivity: {self._totalLinks / self._numNeurons:.2f}")

    def processInput(self, userInput:str):
        self._awaitingInput = self.tokenize(userInput)
        while len(self._awaitingInput) != 0:
            self.forwardStep()

        # cycle solves issue: don't ask for new input while output is still printing
        oldPermanent = self._lastPermanentActivations
        while True:
            self.forwardStep()
            newPerm = self._activations[0:self._permanentCount].copy()
            if np.array_equal(oldPermanent, newPerm):
                break
            oldPermanent = newPerm

    def printSensors(self):
        """
        Prints sensors in format 'string'->neuronNumber
        Sorted alphabetically by string keys
        """
        print("Sensors:")
        # Sort by keys (strings) alphabetically
        sorted_sensors = sorted(self._sensors.items(), key=lambda x: x[0])
        
        for sensor_key, neuron_num in sorted_sensors:
            print(f"'{sensor_key}'->{neuron_num}")
    
    def printEffectors(self):
        """
        Prints effectors in format 'string'->neuronNumber
        Sorted alphabetically by string values
        """
        print("Effectors:")
        # Sort by values (strings) alphabetically
        sorted_effectors = sorted(self._effectors.items(), key=lambda x: x[1])
        
        for neuron_num, effector_value in sorted_effectors:
            # Handle special characters like newline
            if effector_value == '\n':
                display_value = '\\n'
            else:
                display_value = effector_value
            print(f"'{display_value}'->{neuron_num}")

    def consumeOutputBuffer(self)->str:
        ret = ''.join(self._outputBuffer)
        self._outputBuffer = []
        return ret

if __name__ == "__main__":
    print("Subclass this network and fill with data, look at examples")
    quit(0)