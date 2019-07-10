def Process_and_predict(df,columnname):
    import pickle
    from Standardizer import standardize_text
    #df = df.dropna()
    df = df.reset_index()
    test = df.loc[:,[columnname]]
    
    text = standardize_text(test,columnname)
    
    #loading saved tfidf Function
    tfidf = pickle.load(open('tfidf.sav', 'rb'))
    text = tfidf.transform(text[columnname])
    
    #Loading the saved model 
    filename = 'sentiment_model.sav'
    clf = pickle.load(open(filename, 'rb'))
    
    
    Prediction_proba = clf.predict_proba(text,)
    test['proba_0'] = Prediction_proba[:,0]
    test['proba_1'] = Prediction_proba[:,1]
    df['comp_sentiment'] = "Neutral"
    df.loc[test.proba_1>=0.55,"comp_sentiment"] = "Positive"
    df.loc[test.proba_1<=0.35,"comp_sentiment"] = "Negative"
    
    return df