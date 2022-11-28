# imports
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import pandas as pd
import pickle

# read the data
card_df = pd.read_csv('card_transdata.csv')
card_df = card_df.astype({'repeat_retailer': int, 'used_chip': int,
                         'used_pin_number': int, 'online_order': int, 'fraud': int})

# downsample card_df
not_fraud = card_df[card_df.fraud == 0]
fraud = card_df[card_df.fraud == 1]

not_fraud = not_fraud.sample(n=5000, random_state=1)
fraud = fraud.sample(n=5000, random_state=1)

card_df_resampled = pd.concat([not_fraud, fraud])

# split X and y
X = card_df_resampled[card_df_resampled.columns.difference(['fraud'])]
y = pd.DataFrame(card_df_resampled['fraud'])

# split training and testing data
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=7,
                                                    stratify=y
                                                    )

# classifier
classifier = SVC(C=1000, kernel='rbf', random_state=15)

# fit the model
classifier.fit(X_train, y_train)

# Saving model to disk
pickle.dump(classifier, open('model.pkl','wb'))

# loading model to compare the results
model = pickle.load(open('model.pkl','rb'))

# predict
y_pred = model.predict(X_test)
print("Predicted Fraud Recall:", classification_report(
    y_test, y_pred, output_dict=True)['1']['recall'])
print("Accuracy:", classification_report(
    y_test, y_pred, output_dict=True)['accuracy'])
print("\n")