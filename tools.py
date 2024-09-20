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

def denormalize(data, axis, mean = None, std = None):
    #denormalize the data to original distribution
    if mean is None:
        mean = 0
    if std is None:
        std = 1
    return data * std + mean

def littlefr(x, y):
    x = np.array(x)
    y = np.array(y)
    return x @ y / x @ x

def bigFR(x, y):
    x = np.array(x)
    y = np.array(y)
    return x @ y / np.sqrt(x @ x)


#second to hour:min:00
def sec_to_hms(sec):
    minute = int(sec) // 60
    hour = minute // 60
    minute = minute % 60
    #padding 0
    return str(hour).zfill(2) + ':' + str(minute).zfill(2) + ':00'

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

def groupbyday(pred, ground, days):

    #pred and ground are both np.array
    #days stores the days of the data

    #each row of days represents the date of the data
    #group the data by date
    pred, ground, days = np.array(pred), np.array(ground), np.array(days)
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
    return pred_group, ground_group, corr
    #the correlation between pred and ground every day

def sharpeyear(corr, days):
    #days is in the form of a unsorted list, each like '2016-01-01'
    #corr is the correlation between pred and ground every day
    #calculate the sharpe ratio of the year

    #sort the days
    days = sorted(days)
    #calculate the sharpe ratio of every year
    sharpe = []
    lastyear = 0
    for i in range(len(days) - 1):
        if days[i][:4] != days[i + 1][:4]:
            sharpe.append(np.mean(corr[lastyear : i + 1]) / np.std(corr[lastyear : i + 1]))

    return sharpe

def remore_parentheses(string):
    #extract the string in the parentheses
    if "(" not in string:
        return string
    return string[string.find("(") + 1 : string.find(")")]

#format LQ_tdvol_so_wgt_exec
def phrase(string):
    data = string.split('_')
    returns = []
    index = 2
    #check if so or Ts
    if data[index] == 'so':
        returns.append('so')
    else:
        returns.append('Ts')
    index += 1
    #check if wgt or ''
    if data[index] == 'wgt':
        returns.append('wgt')
        index += 1
    else:
        returns.append('')
    #check if exec or ''
    if data[index] == 'exec':
        returns.append('exec')
    else:
        returns.append('')
    
    return data
