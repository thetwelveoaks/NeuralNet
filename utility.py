def convert(num):
    try:
        return float(num)
    except ValueError:
        return int(num)

def readfile(filename):
    f = open(filename, 'r')
    dataset = []
    for line in f:
        line = line.rstrip('\n').split(',')
        inputs = []
        for num in line:
            inputs.append(convert(num))
        dataset.append(inputs)
        
    data_pairs = []
    for data in dataset:
        data_pairs.append([data[:-1]] + [data[-1]])
    return data_pairs

def splitdata(dataset):
    import random
    seed = 123456
    random.seed(seed)
    random.shuffle(dataset)
    dataset_cpy = dataset
    training = dataset_cpy[0: int(0.7 * len(dataset)) + 1]
    dataset_cpy = dataset[int(0.7 * len(dataset)) + 1:]
    validation = dataset_cpy[0: int(0.5 * len(dataset_cpy)) + 1]
    testing = dataset_cpy[int(0.5 * len(dataset_cpy)) + 1:]
    return [training, validation, testing]

def normalise(dataset):
    mini = min(dataset[0])
    maxi = max(dataset[0])
    return [[(num - mini) / (maxi - mini) for num in dataset[0]]] + [dataset[1]]


dataset = readfile('spambase.data')
splited = splitdata(dataset)

training = [normalise(item) for item in splited[0]]
validation = [normalise(item) for item in splited[1]]
testing = [normalise(item) for item in splited[2]]

#print(training[0])
#training = [normalise(item) for item in training]

#normalise(training[0])

#print(training[0])
#print(validation[0])
#print(testing[0])


