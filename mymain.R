### Read Data ###
all = read.table("data.tsv", stringsAsFactors = F,header = T)
all$review = gsub('<.*?>', ' ', all$review)
splits = read.table("splits.csv", header = T)
s = 3

### Packages ###
if (!require("text2vec")) {
  install.packages("text2vec")
}

if (!require("pROC")) {
  install.packages("pROC")
}

if (!require("glmnet")) {
  install.packages("glmnet")
}

library(tidyverse)
library(text2vec)
library(glmnet)
library(pROC)

### Test/Train Split ###
train = all[-which(all$new_id%in%splits[, s]),]
test = all[which(all$new_id%in%splits[, s]),]

### Read Vocab ###
vocab = readRDS("myVocab.rds")

prep_fun = tolower
tok_fun = word_tokenizer

it_train = itoken(train$review, 
                  preprocessor = prep_fun, 
                  tokenizer = tok_fun, 
                  ids = train$new_id, 
                  progressbar = FALSE)

vectorizer = vocab_vectorizer(vocab)
dtm_train = create_dtm(it_train, vectorizer)

tfidf = TfIdf$new()

# fit model to train data and transform train data with fitted model
dtm_train_tfidf = fit_transform(dtm_train, tfidf)

### Test stuff ###
it_test = itoken(test$review, 
                  preprocessor = prep_fun, 
                  tokenizer = tok_fun, 
                  ids = test$new_id, 
                  progressbar = FALSE)

dtm_test = create_dtm(it_test, vectorizer)
dtm_test_tfidf = transform(dtm_test, tfidf)

### Model ###
NFOLDS = 10
set.seed(500)
ytrain = train$sentiment
mycv = cv.glmnet(x=dtm_train_tfidf, y=ytrain, 
                 family='binomial',type.measure = "auc", 
                 nfolds = NFOLDS, alpha=0)

myfit = glmnet(x=dtm_train_tfidf, y=ytrain, 
               lambda = mycv$lambda.min, family='binomial', alpha=0)

pred = round(predict(myfit, dtm_test_tfidf, type = "response"), 2)
roc_obj = roc(test$sentiment, as.vector(pred))
auc = pROC::auc(roc_obj) 

out = cbind(test$new_id, pred)
write.table(out, file = paste("Result_", s, ".txt", sep = ""), col.names = c("new_id", "prob"), 
            row.names = F, sep = ", ", quote = F)

