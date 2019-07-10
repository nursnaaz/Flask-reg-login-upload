def preprocess_text(df,columnname):
    import warnings
    import nltk
    from nltk import FreqDist
    nltk.download('punkt')
    import pandas as pd
    nltk.download('stopwords')
    warnings.filterwarnings("ignore")
    import re
    df = df.drop_duplicates(subset =columnname)
    print(df.shape)
    df[columnname] = df[columnname].map(lambda x: re.sub(r'http\S+', '', str(x)))
    df[columnname] = df[columnname].map(lambda x: re.sub(r'[^ a-zA-Z0-9!?:,.\'=]', '', str(x)))
    df[columnname] = df[columnname].str.lower()
    # Expand contractions
    import re
    contractions_dict = {
        'didn\'t': 'did not',
        'don\'t': 'do not',
        "aren't": "are not",
        "can't": "cannot",
        "cant": "cannot",
        "can't've": "cannot have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "didnt": "did not",
        "doesn't": "does not",
        "doesnt": "does not",
        "don't": "do not",
        "dont" : "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he had",
        "he'd've": "he would have",
        "he'll": "he will",
        "he's": "he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how is",
        "i'd": "i had",
        "i'd've": "i would have",
        "i'll": "i will",
        "i'm": "i am",
        "im": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'll": "it will",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she had",
        "she'd've": "she would have",
        "she'll": "she will",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "that's": "that is",
        "there's": "there is",
        "they'd": "they had",
        "they'd've": "they would have",
        "they'll": "they will",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we had",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where is",
        "where've": "where have",
        "who'll": "who will",
        "who's": "who is",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "you'll": "you will",
        "you're": "you are",
        "you've": "you have"
        }

    contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))
    def expand_contractions(s, contractions_dict=contractions_dict):
        def replace(match):
            return contractions_dict[match.group(0)]
        return contractions_re.sub(replace, s)
    df[columnname] = df[columnname].apply(expand_contractions)
    df = df.reset_index(drop=False)
    
    data = df[[columnname, 'index']]
    print(data.columns)
    data.rename(columns={'index': 'INDEX'}, inplace=True)
    from nltk.tokenize import sent_tokenize
    data['split'] = data[columnname].apply(sent_tokenize)
    data_split = data.set_index('INDEX').split.apply(pd.Series).stack().reset_index(level=0).rename(columns={0:columnname})
    data_split.reset_index(level=0, inplace=True)
    data_split.rename(columns={'INDEX': 'review_no', 'index': 'sentence'}, inplace=True)
    # Spell Correct Algorithm
    from symspellpy.symspellpy import SymSpell  # import the module
    max_edit_distance_dictionary = 0
    prefix_length = 7
    # create object
    sym_spell = SymSpell(max_edit_distance_dictionary, prefix_length)
    dictionary_path = "./frequency_dictionary_en_82_765.txt"
    term_index = 0  # column of the term in the dictionary text file
    count_index = 1  # column of the term frequency in the dictionary text file
    #if not sym_spell.load_dictionary(dictionary_path, term_index, count_index):
    sym_spell.load_dictionary(dictionary_path, term_index, count_index)
    data_split[columnname] = data_split[columnname].apply(sym_spell.word_segmentation)
    data_split[columnname] = data_split[columnname].apply(lambda x: x.corrected_string)
    from Sentiment_prediction_noor import Process_and_predict
    data_split = Process_and_predict(data_split,columnname)
    
    df.rename(columns={'index': 'review_no'}, inplace=True)
    
    #print(df.head())
    df.rename(columns = {columnname: "Text"},inplace = True)
    data_final = data_split.merge(df, on="review_no", how = 'left')
    print(data_final.columns)
    #print(data_final.head(2))
    # Natural Language Processing of Reviews (Top Features, Feelings, Actions)
    import spacy
    from spacy import displacy
    from collections import Counter
    import en_core_web_sm
    nlp = en_core_web_sm.load()
    from nltk.corpus import stopwords 
    stopwords = stopwords.words('english')
    newStopWords = ['-PRON-']
    stopwords.extend(newStopWords)

    words2 = ['no','nor','not']
    for word in list(stopwords):  # iterating on a copy since removing will mess things up
        if word in words2:
            stopwords.remove(word)
    # remove short words (length =< 0) 
    final = data_final[columnname].apply(lambda x: ' '.join([w for w in x.split() if len(w)>0]))
    final = pd.DataFrame(final)
    #tokenized_reviews = pd.Series(reviews).apply(lambda x: x.split())
    tokenized_reviews = pd.Series(final[columnname]).apply(lambda x: x.split())
    #def lemmatization(texts, tags=['NOUN', 'ADJ']):
    #def lemmatization(texts, tags=['NOUN', 'ADJ', 'VERB']):
    def lemmatization(texts, tags=['NOUN']):
        output = []        
        for sent in texts:
            doc = nlp(" ".join(sent))                             
            output.append([token.lemma_ for token in doc if 
            token.pos_ in tags])        
        return output

    def lemmatization_adj(texts, tags=['ADJ']):
        output = []        
        for sent in texts:
            doc = nlp(" ".join(sent))                             
            output.append([token.lemma_ for token in doc if 
            token.pos_ in tags])        
        return output

    def lemmatization_verb(texts, tags=['VERB']):
        output = []        
        for sent in texts:
            doc = nlp(" ".join(sent))                             
            output.append([token.lemma_ for token in doc if 
            token.pos_ in tags])        
        return output
    
    noun_adj_pairs = []
    noun_adj_sent = []
    import spacy
    for l in range(len(final)):
        doc = nlp(str(final[columnname][l]))
        noun_adj_pairs = []
        for i,token in enumerate(doc):
            #print(token)
            if token.pos_ not in ('NOUN','PROPN'):
                continue
            for j in range(i+1,len(doc)):
                if doc[j].pos_ == 'ADJ':
                    noun_adj_pairs.append((token,doc[j]))
                    break
        noun_adj_sent.append(noun_adj_pairs)
    final['noun_adj'] = noun_adj_sent
    
    #Noun Extraction
    
    print("Beginning Noun Extraction")
    reviews_noun = lemmatization(tokenized_reviews)

    reviews_3 = []
    for i in range(len(reviews_noun)):
        reviews_3.append(' '.join(reviews_noun[i]))
    final['features'] = reviews_3

    print("Noun Extraction Complete")
    
    #Adjective Extraction 
    
    print("Beginning Adjectives Extraction")
    reviews_adj = lemmatization_adj(tokenized_reviews)

    reviews_4 = []
    for i in range(len(reviews_adj)):
        reviews_4.append(' '.join(reviews_adj[i]))
    final['feelings'] = reviews_4

    print("Adjectives Extraction Complete")
    
    #Verb Extraction
    
    print("Beginning Verb Extraction")
    reviews_verb = lemmatization_verb(tokenized_reviews)

    reviews_5 = []
    for i in range(len(reviews_verb)):
        reviews_5.append(' '.join(reviews_verb[i]))
    final['action'] = reviews_5

    print("Verb Extraction Complete")
    
    # remove short words (length =< 3) 
    final['features'] = final['features'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))
    final['feelings'] = final['feelings'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))
    final['action']   = final['action'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))
    #final['adverb']   = final['adverb'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))
    
    # Remove Stopwords and Spaces (Start and End)
    def remove_stopwords(text):
        text = ' '.join([word for word in text.split() if word not in stopwords])
        return text.strip()
    final['features'] = final['features'].apply(remove_stopwords)
    final['feelings'] = final['feelings'].apply(remove_stopwords)
    final['action'] = final['action'].apply(remove_stopwords)
    
    final = final.drop([columnname], axis=1)
    def freq_words(x): 
        all_words = ' '.join([text for text in x]) 
        all_words = all_words.split() 
        #print(all_words)
        fdist = FreqDist(all_words) 
        words_df = pd.DataFrame({'word':list(fdist.keys()),'count':list(fdist.values())})
        # selecting top 20 most frequent words 
        #d = words_df.nlargest(columns="count", n = length)

        # take words that covers up to 80th quantile 
        length = int(words_df['count'].quantile(0.85))
        #print(length)
        # sort results - descending
        d = words_df[words_df['count'] >= length].sort_values(['count'], ascending = False)
        d = d['word'].tolist()
        return d

    feature_filt_dict = freq_words(final['features'])
    action_filt_dict = freq_words(final['action'])
    feeling_filt_dict = freq_words(final['feelings'])
    
    def remove_min_words(text, dictionary):
        text = ' '.join([word for word in text.split() if word in dictionary])
        return text

    final['feelings'] = final['feelings'].apply(remove_min_words, dictionary = feeling_filt_dict)
    final['action'] = final['action'].apply(remove_min_words, dictionary = action_filt_dict)
    final['features'] = final['features'].apply(remove_min_words, dictionary = feature_filt_dict)
    #final['adverb'] = final['adverb'].apply(remove_min_words, dictionary = feature_filt_dict)
    
    # Remove duplicate words in cell
    from collections import OrderedDict

    final['features'] = final['features'].str.split().apply(lambda x:OrderedDict.fromkeys(x).keys()).str.join('')
    final['feelings'] = final['feelings'].str.split().apply(lambda x:OrderedDict.fromkeys(x).keys()).str.join('')
    final['action']   = final['action'].str.split().apply(lambda x: OrderedDict.fromkeys(x).keys()).str.join(' ')
    #final['adverb']   = final['adverb'].str.split().apply(lambda x: OrderedDict.fromkeys(x).keys()).str.join('')
    print(final.columns)
    rows = []
    _ = final.apply(lambda row: [rows.append([row['features'],row['feelings'],row['action'],na]) for na in row.noun_adj], axis=1)
    final = pd.DataFrame(rows, columns=['features', 'feelings', 'action','noun_adj'])
    
    #pd.concat([df_new.noun_adj.str.extract('(?P<col1>\d+),(?P<col2>\d+)'),df_new], axis = 1)
    final = pd.concat([final.noun_adj.apply(lambda x: pd.Series(x, index=['Feature_n', 'Feeling_adj'])), 
          final], axis=1)
    
    final_data = pd.concat([data_final, final], axis = 1)
    
    final_data.reset_index(inplace=True)
   
    
    print(final_data.head())
    print(final_data.dtypes)
    
    
    ## Keep Reviews that are 3 characters or longer
    #final_data = final_data[final_data['Text'].apply(lambda x: len(x) > 3)]
    mask = (final_data['Text'].str.len() > 3)
    final_data = final_data.loc[mask]
    print(final_data.columns)
    final_data = final_data.drop(columns = 'index')
    
    mycolumns = final_data.columns
    data_final_nlp = final_data[mycolumns]

    features_file = data_final_nlp[['sentence','review_no',columnname,'Text','comp_sentiment','features']]

    actions_file = data_final_nlp[['sentence','review_no',columnname,'Text','comp_sentiment','action']]

    feelings_file = data_final_nlp[['sentence','review_no',columnname,'Text','comp_sentiment','feelings']]
    
    noun_adj_file = data_final_nlp[['sentence','review_no',columnname,'Text','comp_sentiment','Feature_n', 'Feeling_adj']]
    # Explode pandas dataframe string entry to separate rows
    import pandas as pd
    import numpy as np

    def explode(df, lst_cols, fill_value='', preserve_index=False):
        # make sure `lst_cols` is list-alike
        if (lst_cols is not None
            and len(lst_cols) > 0
            and not isinstance(lst_cols, (list, tuple, np.ndarray, pd.Series))):
            lst_cols = [lst_cols]
        # all columns except `lst_cols`
        idx_cols = df.columns.difference(lst_cols)
        # calculate lengths of lists
        lens = df[lst_cols[0]].str.len()
        # preserve original index values    
        idx = np.repeat(df.index.values, lens)
        # create "exploded" DF
        res = (pd.DataFrame({
                    col:np.repeat(df[col].values, lens)
                    for col in idx_cols},
                    index=idx)
                 .assign(**{col:np.concatenate(df.loc[lens>0, col].values)
                                for col in lst_cols}))
        # append those rows that have empty lists
        if (lens == 0).any():
            # at least one list in cells is empty
            res = (res.append(df.loc[lens==0, idx_cols])
                      .fillna(fill_value))
        # revert the original index order
        res = res.sort_index()
        # reset index if requested
        if not preserve_index:        
            res = res.reset_index(drop=True)
        return res
    
    #Creation of Features, Feelings and Actions File
    features_file = features_file.assign(features=features_file.features.str.split(' '))
    actions_file = actions_file.assign(action=actions_file.action.str.split(' '))
    feelings_file = feelings_file.assign(feelings=feelings_file.feelings.str.split(' '))
    #adverb_file = adverb_file.assign(adverb=adverb_file.adverb.str.split(' '))
    
    features_file = explode(features_file, ['features'], fill_value='')
    actions_file = explode(actions_file, ['action'], fill_value='')
    feelings_file = explode(feelings_file, ['feelings'], fill_value='')
    #adverb_file = explode(adverb_file, ['adverb'], fill_value='')
    
    features_file.to_excel("Features_file.xlsx",index = False)
    actions_file.to_excel("Actions_file.xlsx",index = False)
    feelings_file.to_excel("Feelings_file.xlsx",index = False)
    noun_adj_file.to_excel("Noun_adj_file.xlsx",index = False)
    
    return features_file,actions_file,feelings_file,noun_adj_file