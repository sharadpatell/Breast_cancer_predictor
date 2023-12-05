import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
# import pickle5 as pickle
import pickle

def create_model(data):
    X = data.drop(['diagnosis'], axis = 1)
    y = data['diagnosis']

    #scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    #split the data into train test
    X_train, X_test, y_train, y_test = train_test_split(
        X,y, test_size=0.2,random_state=42
    )

    #train
    model = LogisticRegression()
    model.fit(X_train, y_train)

    #test model 
    y_pred = model.predict(X_test)
    print("Accuracy of our model:", accuracy_score(y_test, y_pred))
    print("Classification report: \n",classification_report(y_test, y_pred))

    return model, scaler


def get_clean_data():
    data = pd.read_csv("data/data.csv")
    
    #drop the irrevelant columns 
    data = data.drop(['id','Unnamed: 32'], axis=1)

    #convert diagnosis into 0 and 1
    data['diagnosis'] = data['diagnosis'].map({'M':1, 'B':0})

    return data



def main():
    data = get_clean_data()
    
    model, scaler = create_model(data) 

    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)    


if __name__ == '__main__':
    main()