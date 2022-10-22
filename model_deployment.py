import joblib
model = joblib.load("Email_Spam_Detection_model.joblib")
vectorizer = joblib.load("vectorizer.joblib")
input_data = input("Enter the email : ")
# input_data = ["""WINNER!! As a valued network customer you have been selected 
#               to receivea Â£900 prize reward! To claim call 09061701461. Claim 
#               code KL341. Valid 12 hours only."""]
input_data_features = vectorizer.transform(input_data)
prediction = model.predict(input_data_features)

'''
  0 ----> spam
  1 ----> Ham
'''
if (prediction[0]==0):
  print('Ham mail')

else:
  print('Spam mail')