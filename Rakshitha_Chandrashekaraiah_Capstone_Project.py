import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score


def read_CSV(url):
    #function to read a CSV file
    crime = pd.read_csv(url)
    return crime


def output_Graph(crime,x_value):
    #funtion to produce graph
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 10))
    sns.catplot(x=x_value,
                kind='count',
                color='Orange',
                data=crime[(crime.Shooting == 'Y')], ax=axes[0])
    axes[0].set_xlabel('{}'.format(x_value), fontsize=12)
    axes[0].set_ylabel('Number of Shooting', fontsize=12)
    axes[0].set_title('Number of Shooting Incidents based on {}'.format(x_value), fontsize=14)
    plt.close()

    sns.catplot(x=x_value,
                kind='count',
                color='Orange',
                data=crime, ax=axes[1])
    axes[1].set_xlabel('{}'.format(x_value), fontsize=12)
    axes[1].set_ylabel('Number of Criminal Incidents', fontsize=12)
    axes[1].set_title('Number of Criminal Incidents based on {}'.format(x_value), fontsize=14)
    plt.close()
    return fig


def output_District(crime):
    #function to replace District codes with Distric Names
    crime.District.replace('C11', 'Dorchester', inplace=True)
    crime.District.replace('E13','Jamaica Plain', inplace=True)
    crime.District.replace('E5', 'West Roxbury', inplace=True)
    crime.District.replace('B2', 'Roxbury', inplace=True)
    crime.District.replace('D4', 'South End', inplace=True)
    crime.District.replace('A1', 'Downtown', inplace=True)
    crime.District.replace('B3', 'Mattapan', inplace=True)
    crime.District.replace('D14', 'Brighton', inplace=True)
    crime.District.replace('C6', 'South Boston', inplace=True)
    crime.District.replace('A15', 'Charlestown', inplace=True)
    crime.District.replace('E18', 'Hyde Park', inplace=True)
    crime.District.replace('A7', 'East Boston', inplace=True)
    return crime


def fill_Nan(crime_data):
    #Funtion to fill Nan
    crime_data["SHOOTING"].fillna("N", inplace=True)
    crime_data["STREET"].fillna("UNKNOWN", inplace=True)
    crime_data["DISTRICT"].fillna("Unknown", inplace=True)
    crime_data["UCR_PART"].fillna("Unknown", inplace=True)
    crime_data['Lat'].fillna(-1, inplace=True)
    crime_data['Long'].fillna(-1, inplace=True)
    crime_data.Lat.replace(-1, None, inplace=True)
    crime_data.Long.replace(-1, None, inplace=True)
    return crime_data


def calc_prevalence(y_actual):
    #function to calculate the prevalence of the positive class
    return (sum(y_actual)/len(y_actual))


def calc_specificity(y_actual, y_pred, thresh):
    #funtion to calculate specificity
    return sum((y_pred < thresh) & (y_actual == 0)) / sum(y_actual == 0)


def print_report(y_actual, y_pred, thresh):
    #funtion to find performance measures
    auc = roc_auc_score(y_actual, y_pred)
    accuracy = accuracy_score(y_actual, (y_pred > thresh))
    recall = recall_score(y_actual, (y_pred > thresh))
    precision = precision_score(y_actual, (y_pred > thresh))
    specificity = calc_specificity(y_actual, y_pred, thresh)
    print('AUC:%.3f' % auc)
    print('accuracy:%.3f' % accuracy)
    print('recall:%.3f' % recall)
    print('precision:%.3f' % precision)
    print('specificity:%.3f' % specificity)
    print('prevalence:%.3f' % calc_prevalence(y_actual))
    print(' ')
    return auc, accuracy, recall, precision, specificity
