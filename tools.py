import numpy as np
#data is in the form of a DataFrame
def normalize(data, axis, mean = None, std = None):
    #normalize the data to normal distribution
    #axis = 0 means normalize each column
    #axis = 1 means normalize each row
    #mean and std are the mean and standard deviation of the data
    #if mean and std are not given, calculate them from the data
    if mean is None:
        mean = data.mean(axis = axis)
    if std is None:
        std = data.std(axis = axis)
    return (data - mean) / std, mean, std

def notnulldataframe(data):
    #return the rows of data that have no null values
    return data[~data.isnull().any(axis = 1)]

def notnullobj(data):
    #return the rows of data that have no null values
    return data[~data.isnull()]

all = np.array([])
#append the pd objects to all
def append(obj):
    global all
    all = np.append(all, obj)


#save np.array into a csv file
def savecsv(filename):
    global all
    np.savetxt(filename, all, delimiter = ',', fmt = '%s')
    all = np.array([])

#np.array to dataframe
def todf():
    global all
    return pd.DataFrame(all)

def correlation(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
    r = Ridge()
    r.fit(x_train, y_train)
    print(r.score(x_train, y_train))
    print(r.score(x_test, y_test))
    return np.corrcoef(r.predict(x_test), y_test)[0, 1]