import time, math, pdb
import numpy as np
import pandas as pd
from collections import defaultdict
import sys
import string

def log(color=None, message=None, messageType="" , endChar="\n", alt = True, nl = 1):
    if color == None:
        for i in range(nl):
            print()
        return
    colors = {
    'HEADER': '\033[95m',
    'OKBLUE': '\u001b[38;5;6m',
    'OKGREEN': '\033[92m',
    'WARNING': '\033[93m',
    'FAIL': '\033[91m',
    'ENDC': '\033[0m',
    'BOLD': '\033[1m',
    'UNDERLINE': '\033[4m',
    'BLACK': '\u001b[38;5;0m'}
    if messageType != "":
        messageType = f"({messageType})  "
    if not alt:
        print(f"{colors[color]}{messageType}{message}{colors['ENDC']}", end = endChar, flush=True)
    else:
        print(f"{colors[color]}{messageType}{colors['ENDC']}{colors['BLACK']}{message}{colors['ENDC']}", end = endChar, flush=True)
        
Activations = {
    "sigmoid": {"activation": lambda x: 1 / (1 + math.exp(-1*x)), "dactivation": lambda x: (1 / (1 + math.exp(-1*x)))*(1-(1 / (1 + math.exp(-1*x))))},
    "tanh": {"activation": lambda x: np.tanh(x), "dactivation": lambda x: 1 - (np.tanh(x))**2},
    "relu": {"activation": lambda x: max(0, x), "dactivation": lambda x: 1 if x>0 else 0},
}

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

    
class Node:
    numNodes = 0
    def __init__(self, activation, indegree, outdegree, name=f"GC", simulationSteps = []):
        '''
        Constructor for Node
        
        Params:
        activation (string) - activation type to be used
        indegree (list of Edge) - Edges pointing into the node
        outdegree (list of Edge) - Edges pointing out of the node
        name (string) - unique name/identifier for node
        simulationSteps (list of floats) - past simulation values, if any
        
        Return:
        N/A
        '''
        if simulationSteps == []:
            self.simulationSteps = [0.0]
        else:
            self.simulationSteps = simulationSteps[:]                
        self.activation = activation
        self.indegree = indegree
        self.outdegree = outdegree
        self.name = name
        Node.numNodes += 1
    
    def getIndegreeNodes(self):
        '''
        Getter generator for nodes attached to indegree edges
        
        Params:
        N/A
        
        Yields:
        (node): nodes connected to indegree edges
        '''
        for edge in self.indegree:
            yield edge.origin
            
    def getOutdegreeNodes(self):
        '''
        Getter generator for nodes attached to outdegree edges
        
        Params:
        N/A
        
        Yields:
        (node): nodes connected to outdegree edges
        '''
        for edge in self.outdegree:
            yield edge.destination

    def updateValue(self, value):
        '''
        Update a value of a node
        
        Params:
        value (float): input to a node (pre-activation)
        
        Return:
        N/A
        '''
        self.simulationSteps.append(Activations[self.activation]['activation'](value))
    
    def dryValue(self, value):
        '''
        Dry update a value of a node
        
        Params:
        value (float): input to a node (pre-activation)
        
        Return:
        (float): activated result
        '''
        return Activations[self.activation]['activation'](value)
    
    def __str__(self):
        return f"{self.name}\n\tCurrent: {self.simulationSteps[-1]}\n\tIndegree: {self.indegree}\n\tOutdegree: {self.outdegree}\n\tHistory: {self.simulationSteps[:-1]}\n"
       
    def __repr__(self):
        return self.__str__()
     
class Edge:
    def __init__(self, origin, destination, weight = 0.5, buffer = 0):
        '''
        Constructor for Edge
        
        Params:
        origin (Node): origin node of edge (connected to tail of edge)
        destination (Node): destination node of edge (connected to head of edge)
        weight (float): magnitude of causality
        buffer (N/A): NOT IMPLEMENTED
        
        Return:
        N/A
        '''
        self.weight = weight
        self.buffer = buffer
        self.origin = origin
        self.destination = destination
    
    def __str__(self):
        return f"Weight ({self.origin.name} -> {self.destination.name}): {self.weight}"
    
    def __repr__(self):
        return self.__str__()

class Graph:
    def __init__(self, nodes = [], weightMatrix=None, activation='sigmoid'):
        '''
        Constructor for Graph (combination of Edges and Nodes)
        
        Params:
        nodes (list of Nodes): list of nodes if pre-created
        weightMatrix (Node): Pandas DataFrame consisting of a weight matrix of weight values where rows are origin node names and columns are destination node names

        Return:
        N/A
        '''
        if type(weightMatrix) != "None":
            nodes = defaultdict(lambda: '')
            for i, index in enumerate(weightMatrix.index):
                nodes[index] = Node(activation, [], [], index)
            for a, index in enumerate(weightMatrix.index):
                for b, column in enumerate(weightMatrix.columns):
                    if not math.isnan(weightMatrix.iloc[a][b]):
                        weight = Edge(nodes[index], nodes[column], weightMatrix.iloc[a][b])
                        nodes[index].outdegree.append(weight)
                        nodes[column].indegree.append(weight)
            self.nodes = list(nodes.values())
        else:
            self.nodes = nodes
            
    def step(self):
        '''
        Runs an iteration of graph simulation
        
        Params:
        N/A
        
        Return:
        (Pandas Series): Series of Node values from after simulation step
        '''
        index = len(self.nodes[0].simulationSteps)-1
        res = pd.Series(index = [node.name for node in self.nodes], dtype="float64")
        for node in self.nodes:
            vals = np.array([item.simulationSteps[index] for item in node.getIndegreeNodes()])
            weights = np.array([item.weight for item in node.indegree])
            node.updateValue(sum(np.multiply(vals, weights)))
            res[node.name] = node.simulationSteps[-1]
        return res
    
    def dryStep(self, inputs):
        '''
        Dry runs an iteration of graph simulation
        
        Params:
        N/A
        
        Return:
        (Pandas Series): Series of Node values from after simulation step
        '''
        res = pd.Series(index = [node.name for node in self.nodes], dtype="float64")
        for node in self.nodes:
            vals = np.array([inputs[item.name] for item in node.getIndegreeNodes()])
            weights = np.array([item.weight for item in node.indegree])
            res[node.name] = node.dryValue(sum(np.multiply(vals, weights)))
        return res

    def runSteps(self, numSteps, probability=-1, numNodes=0):
        '''
        Runs multiple iterations of graph simulation with randomization
        
        Params:
        numSteps (integer): number of steps to simulate
        probability (float): float between 0 and 1 for chance of randomizing nodes at any given step
        numNodes (int): number of node values to randomize if randomization occurs
        
        Return:
        N/A
        '''
        for i in range(numSteps):
            self.step()
            if np.random.random() < probability and len(self.nodes[0].simulationSteps)>0:
                for _ in range(numNodes):
                    self.nodes[np.random.randint(0,len(self.nodes))].simulationSteps[-1] = np.random.random()

            
    def getWeightMatrix(self):
        '''
        Getter method for current Edge matrix
        
        Params:
        N/A
        
        Return:
        (Pandas DataFrame): Edge matrix of Graph
        '''
        weights = [weight for node in self.nodes for weight in node.indegree]
        df = pd.DataFrame( index = [node.name for node in self.nodes], columns = [node.name for node in self.nodes])
        for weight in weights:
            df.loc[weight.origin.name][weight.destination.name] = weight.weight
        return df
    
    def getWeights(self):
        '''
        Getter method for Edges connecting Nodes
        
        Params:
        N/A
        
        Return:
        (List of Edges): Edges used in Graph
        '''
        return [weight for node in self.nodes for weight in node.indegree]
    
    def getNodes(self):
        '''
        Getter method for Nodes in Graph
        
        Params:
        N/A
        
        Return:
        (Pandas DataFrame): DataFrame consisting of Node names as column heads and each row is a simulation step
        '''
        df = pd.DataFrame(index = list(range(len(self.nodes[0].simulationSteps))), columns = [node.name for node in self.nodes])
        for node in self.nodes:
            df[node.name] = node.simulationSteps
        return df
    
    def __str__(self):
        res = ""
        for node in self.nodes:
            res += str(node) + "\n\n"
        return res
    
    def __repr_s_(self):
        return self.__str__()

class Training:
    def runHL(graph, data, learningParam=2):
        '''
        Runs Hebbian Learning on Graph for number of data samples
        Note: UTTERLY TERRIBLE TRAINING ALGO (either that or my implementation is faulty,,, definitely the former)
        
        Params:
        graph (Graph): Graph of Nodes and Edges
        data (Pandas DataFrame): DataFrame of training data with Node names as column heads and each row representing an additional sample
        learningParam (float): hyperparameter for training
        
        Return:
        N/A
        '''
        learningParam = data.shape[0]*3.37
        learningCoeff = lambda t: 0.1*(1-(t/(1.1*learningParam))) 
        learningCoeff = lambda t: 0.05

        nodes = graph.nodes
        assert len(data.columns) == len(nodes), "Invalid input data dimensions."

        for i in range(1, data.shape[0],2):
            dx = data.iloc[i] - data.iloc[i-1]
            weights = graph.getWeights()
            for weight in weights:
                if round(dx[weight.origin.name],6) != 0:
                    weight.weight += (learningCoeff(i)*((dx[weight.origin.name]*dx[weight.destination.name])-weight.weight))

    def runCustom(graph, data, learningRate=0.1):
        '''
        Testing an idea out (didn't work)
        
        Params:
        graph (Graph): Graph of Nodes and Edges
        data (Pandas DataFrame): DataFrame of training data with Node names as column heads and each row representing an additional sample
        learningParam (float): hyperparameter for training
        
        Return:
        N/A
        '''
        nodes = graph.nodes
        assert len(data.columns) == len(nodes), "Invalid input data dimensions."
        for i in range(1, data.shape[0],2):
            dx_actual = data.iloc[i] - data.iloc[i-1]
            print(data.iloc[[i-1,i]])
            print("dry")
            print(pd.concat([data.iloc[i-1],graph.dryStep(data.iloc[i-1])], axis=1).reset_index().T)
            dx_predict = graph.dryStep(data.iloc[i-1]) - data.iloc[i-1]
            weights = graph.getWeights()
            for weight in weights:
                a= dx_actual[weight.destination.name]/dx_actual[weight.origin.name]
                b= dx_predict[weight.destination.name]/dx_predict[weight.origin.name]
                weight.weight += (b-a)*0.05
            print(graph.getWeightMatrix())
            print("\n\n")
        pass
    
    def runOSDR(graph, data, learningRate=0.2):
        '''
        OSDR = One Step Delta Rule (creds go to M. Gregor and P.P. Groumpos)
        A supervised algorithm that applies gradient descent to FCM (no backprop)
        
        Params:
        graph (Graph): Graph of Nodes and Edges
        data (Pandas DataFrame): DataFrame of training data with Node names as column heads and each row representing an additional sample
        learningParam (float): hyperparameter for training
        
        Return:
        N/A
        '''
        nodes = graph.nodes
        printProgressBar(0, math.ceil(int(data.shape[0])/2), prefix = 'Progress:', suffix = 'Complete', length = 50)
        for progress, i in enumerate(range(1, data.shape[0],2)):
            weights = graph.getWeights()
            for a, weight in enumerate(weights):
                error = data.iloc[i] - graph.dryStep(data.iloc[i-1]) 
                node = weight.destination
                vals = np.array([data.iloc[i-1][item.name] for item in node.getIndegreeNodes()])
                weightsarr = np.array([item.weight for item in node.indegree])
                tot = sum(np.multiply(vals, weightsarr))
                du = Activations[node.activation]['dactivation'](tot)
                weight.weight += learningRate * (error[node.name] * du) * (data.iloc[i-1][weight.origin.name])
            printProgressBar(progress+1, math.ceil(int(data.shape[0])/2), prefix = 'Progress:', suffix = 'Complete', length = 50)

        log()
    
SAMPLES = 1000
SIZE = 6

log("OKBLUE", "Setup start", "info")
start = time.time()
def namer(i):
    res = []
    al = string.ascii_uppercase
    for a in range(i):
        res.append(al[a%len(al)]+str(math.floor(a/len(al))+1))
    return res
refArr = (np.random.randint(-15, 15+1, SIZE**2)/10).reshape(SIZE,SIZE)
refMatrix = pd.DataFrame(data = refArr,index = namer(SIZE), columns = namer(SIZE))
np.fill_diagonal(refMatrix.values, np.nan)
resArr = (np.zeros(SIZE**2)).reshape(SIZE,SIZE)
resMatrix = pd.DataFrame(data = resArr,index = namer(SIZE), columns = namer(SIZE))
np.fill_diagonal(resMatrix.values, np.nan)

# refArr = [[math.nan,0.5,0.1,],[0.7,math.nan,0.3 ],[0.9,0.2,math.nan]]
# resArr = [[math.nan,0.0,0.0 ],[0.0,math.nan,0.0 ],[0.0,0.0,math.nan]]
# refMatrix = pd.DataFrame(data = refArr,index = list("ABC"), columns = list("ABC"))
# resMatrix = pd.DataFrame(data = resArr,index = list("ABC"), columns = list("ABC"))
refModel = Graph(weightMatrix = refMatrix)
resModel = Graph(weightMatrix = resMatrix)
log("BLACK", "Reference Matrix")
log("BLACK", str(refModel.getWeightMatrix()))
log()
log("BLACK", "Untrained Matrix")
log("BLACK", str(resModel.getWeightMatrix()))
end = time.time()
log("OKBLUE", f"Setup completed in {round((end-start)*1000, 2)} ms", "info")

log(nl=2)

log("OKBLUE", "Training data creation start", "info")
start = time.time()
trainingData = pd.DataFrame(columns = list("ABC"))
np.random.seed(2)
printProgressBar(0, SAMPLES, prefix = 'Progress:', suffix = 'Complete', length = 50)
for i in range(SAMPLES):
    initialVals = [np.random.uniform(0,1) for i in refModel.nodes]
    before = pd.Series(data = initialVals, index = [node.name for node in refModel.nodes], dtype="float64")
    after = refModel.dryStep(before)
    trainingData = trainingData.append([before,after], ignore_index=True)
    printProgressBar(i + 1, SAMPLES, prefix = 'Progress:', suffix = 'Complete', length = 50)
log()
end = time.time()
log("OKBLUE", f"{int(len(trainingData.index)/2)} samples populated in {round((end-start)*1000, 2)} ms", "info")
log(nl=2)

log("OKBLUE", "Training start", "info")
start = time.time()
Training.runOSDR(resModel, trainingData, 0.4)
end = time.time()
log("OKBLUE", f"Training completed in {round((end-start)*1000, 2)} ms", "info")
log(nl=2)

log("OKGREEN", "", "r esults")
log("BLACK", "Reference Matrix")
log("BLACK", str(refModel.getWeightMatrix()))
log()
log("BLACK", "Trained Matrix")
log("BLACK", resModel.getWeightMatrix())

