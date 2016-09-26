import utility
import entity

dataset = utility.readfile('spambase.data')
splited_data = utility.splitdata(dataset)
training = [utility.normalise(item) for item in splited_data[0]]
validation = [utility.normalise(item) for item in splited_data[1]]
testing = [utility.normalise(item) for item in splited_data[2]]

print(training[0])
