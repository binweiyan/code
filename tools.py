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

def sharp(pred, ground, days):
    #calculate the sharpe ratio of the model, pred is the predicted return, ground is the ground truth
    #pred and ground are both np.array
    #days stores the days of the data
    #the sharpe ratio is calculated as follows:
    #the pnl for every day is the sum of the correlation between pred and ground every day
    #the volatility is the std of the correlation between pred and ground every day
    #the sharpe ratio is pnl / volatility and normalized to the number of days in the data

    #each row of days represents the date of the data
    #group the data by date
    pred_group = {}
    ground_group = {}
    for i in range(len(days)):
        if days[i] not in pred_group:
            pred_group[days[i]] = []
            ground_group[days[i]] = []
        pred_group[days[i]].append(pred[i])
        ground_group[days[i]].append(ground[i])
    #for each date, calculate the correlation between pred and ground
    corr = np.array([np.corrcoef(pred_group[day], ground_group[day])[0, 1] for day in pred_group])
    #calculate the sharpe ratio
    return corr.mean() / corr.std() * np.sqrt(len(corr))
    #the correlation between pred and ground every day