# Software for 'A Unified Feature Representation for Lexical Connotations'
Submission to EACL 2021

## Requirements
python                    3.7.6  
pytorch                   1.5.0  
nltk                      3.4.5  
numpy                     1.18.1  
pandas                    0.25.3  
scikit-learn              0.22.1  


## Connotation Embedding
1. Download English-only Conceptnet numberbatch vectors (version 19.08) 
here: https://github.com/commonsense/conceptnet-numberbatch. Put them in the 
`resources/connotations` directory.

2. Make vector files
```angular2html
cd connotations/data_processing
python save_vectors.py -m 1
```

3. Train connotation embedding model  
For exmaple, to train CE+Rel (J):
```angular2html
cd connotations
./train.sh 1 ../../config/connotations/config-BEST.txt
```

4. Evaluate connotation embedding model
For ex
```angular2html
cd connotations
./eval.sh 2 ../../config/connotations/config-BEST.txt
```

## Stance Detection
1. Download Glove vectors from http://nlp.stanford.edu/data/glove.6B.zip  
Keep the 100d vectors.  
Save to `resources/stance/` directory.

2. Make vectors files
```angular2html
cd stance/preprocessing
python save_vectors.py -m 1
```

3. Train the stance model  
Config files for all experiments are provided. For ex:
```angular2html
cd stance
./train.sh ../../config/stance/config-BiC+c-All.txt
```

4. Evaluate a trained model
```angular2html
cd stance
./eval.sh 2 ../../config/stance/config-BiC+c-ALL.txt
```