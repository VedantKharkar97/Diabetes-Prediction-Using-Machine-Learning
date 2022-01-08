from django.shortcuts import render
import pickle


def home(request):
    return render(request, 'home.html')

def predict(request):
    return render(request, 'predict.html')

def getPredictions(num_preg,glucose_conc,diastolic_bp,thickness,insulin,bmi,diab_pred,age):
    model = pickle.load(open('TC127_DiabetesPrediction.sav', 'rb'))
    scaled = pickle.load(open('scaler1.sav', 'rb'))

    prediction = model.predict(scaled.transform([
        [num_preg,glucose_conc,diastolic_bp,thickness,insulin,bmi,diab_pred,age]
    ]))
    
    if prediction == 0:
        return 'no'
    else: 
        return 'yes'
  

def result(request):

    num_preg = float(request.GET['num_preg'])
    glucose_conc = float(request.GET['glucose_conc'])
    diastolic_bp = float(request.GET['diastolic_bp'])
    thickness = float(request.GET['thickness'])
    insulin = float(request.GET['insulin'])
    bmi = float(request.GET['bmi'])
    diab_pred = float(request.GET['diab_pred'])
    age = float(request.GET['age'])
    

    result = getPredictions(num_preg,glucose_conc,diastolic_bp,thickness,insulin,bmi,diab_pred,age)

    if request.method == 'GET':
        return render(request, 'result.html', {"result":result})

def about(request):
    return render(request, 'about.html')


    
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
    # cls = joblib.load('diabetes_model.sav')
    # lis = []
    # lis.append(request.GET['n1'])
    # lis.append(request.GET['n2'])
    # lis.append(request.GET['n3'])
    # lis.append(request.GET['n4'])
    # lis.append(request.GET['n5'])
    # lis.append(request.GET['n6'])
    # lis.append(request.GET['n7'])
    # lis.append(request.GET['n8'])
    # lis.append(request.GET['n9'])
    
    # print(lis)

    # pred = cls.predict([lis])

    # data = pd.read_csv("D:\College\TE COMP\SEMINAR\DEMO\Seminar_Diabetes_Prediction\Diabetes_Prediction\Diabetes_Dataset.csv")
    
    # diabetes_map = {True: 1, False: 0}
    # data['diabetes'] = data['diabetes'].map(diabetes_map)
    
    # data['insulin'] = data['insulin'].replace(0, data['insulin'].mean()) 
    # data['glucose_conc'] = data['glucose_conc'].replace(0, data['glucose_conc'].mean()) 
    # data['diastolic_bp'] = data['diastolic_bp'].replace(0, data['diastolic_bp'].mean()) 
    # data['bmi'] = data['bmi'].replace(0, data['bmi'].mean()) 
    # data['skin'] = data['skin'].replace(0, data['skin'].mean()) 
    # data['thickness'] = data['thickness'].replace(0, data['thickness'].mean())

    # X = data.drop(columns = 'diabetes', axis=1)
    # Y = data['diabetes'] 

    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # scaler.fit(X)
    # standardized_data = scaler.transform(X)
    # X = standardized_data
    # Y = data['diabetes']

    # X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)
    
    
    # #Apply Machine Learning Algorithm
    # svc_scores = []
    # kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    # for i in range(len(kernels)):
    #     svc_classifier = SVC(kernel = kernels[i])
    #     svc_classifier.fit(X_train, Y_train)
    #     svc_scores.append(svc_classifier.score(X_test, Y_test))
    
   
    # knn_scores = []
    # for k in range(1,21):
    #     knn_classifier = KNeighborsClassifier(n_neighbors = k)
    #     knn_classifier.fit(X_train, Y_train)
    #     knn_scores.append(knn_classifier.score(X_test, Y_test))
    
    
    
    # val1 = float(request.GET['n1'])
    # val2 = float(request.GET['n2'])
    # val3 = float(request.GET['n3'])
    # val4 = float(request.GET['n4'])
    # val5 = float(request.GET['n5'])
    # val6 = float(request.GET['n6'])
    # val7 = float(request.GET['n7'])
    # val8 = float(request.GET['n8'])
    # val9 = float(request.GET['n9'])
    
    # v = np.array(['n1','n2','n3','n4','n5','n6','n7','n8','n9'])
    # w = v.astype(np.float64)
    # # a = ["1.1", "2.2", "3.2"]
    # # b = np.asarray(a, dtype=np.float64, order='C')
    # input_data_as_numpy_array = np.asarray(w)
    # input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    # std_data = scaler.transform(input_data_reshaped)
    
    # pred = svc_classifier.predict([[std_data]])
    
    # result2 = ""
    # if pred == 0:
    #     result2 = "Negative"
    # if pred == 1:
    #     result2 = "Positivedown"    



    