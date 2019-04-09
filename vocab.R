library(text2vec)
library(pROC)
library(glmnet)
library(tidyverse)

data = read.table("data.tsv", stringsAsFactors = FALSE, header = TRUE)
data$review = gsub('<.*?>', ' ', data$review)


### Splits ###
splits = read.table("splits.csv", header = T)


### Split 1 ###
train1 = data[-which(data$new_id%in%splits[,1]),]
test1 = data[which(data$new_id%in%splits[,1]),]

### Split 2 ###
train2 = data[-which(data$new_id%in%splits[,2]),]
test2 = data[which(data$new_id%in%splits[,2]),]

### Split 3 ###
train3 = data[-which(data$new_id%in%splits[,3]),]
test3 = data[which(data$new_id%in%splits[,3]),]

### Vocabulary ###
prep_fun = tolower
tok_fun = word_tokenizer

it_train = itoken(train1$review, 
                  preprocessor = prep_fun, 
                  tokenizer = tok_fun, 
                  ids = train3$new_id, 
                  progressbar = FALSE)

stop_words = c("i", "me", "my", "myself", 
               "we", "our", "ours", "ourselves", 
               "you", "your", "yours", 
               "their", "they", "his", "her", 
               "she", "he", "a", "an", "and",
               "is", "was", "are", "were", 
               "him", "himself", "has", "have", 
               "it", "its", "of", "one", "for", 
               "the", "us", "this")

vocab = create_vocabulary(it_train, ngram = c(1L, 2L), stopwords = stop_words)

pruned_vocab = prune_vocabulary(vocab, 
                                term_count_min = 10, 
                                doc_proportion_max = 0.5,
                                doc_proportion_min = 0.001)


### DTM ###
vectorizer = vocab_vectorizer(pruned_vocab)
dtm_train = create_dtm(it_train, vectorizer)



### Modeling ###
v.size = dim(dtm_train)[2]
ytrain = train1$sentiment

summ = matrix(0, nrow=v.size, ncol=4)
summ[,1] = apply(dtm_train[ytrain==1, ], 2, mean)
summ[,2] = apply(dtm_train[ytrain==1, ], 2, var)
summ[,3] = apply(dtm_train[ytrain==0, ], 2, mean)
summ[,4] = apply(dtm_train[ytrain==0, ], 2, var)
n1=sum(ytrain); 
n=length(ytrain)
n0= n - n1

myp = (summ[,1] - summ[,3])/
  sqrt(summ[,2]/n1 + summ[,4]/n0)

words = colnames(dtm_train)
id = order(abs(myp), decreasing=TRUE)[1:3000]
pos.list = words[id[myp[id]>0]]
neg.list = words[id[myp[id]<0]]
txt = c(pos.list, neg.list)
it = itoken(txt, tolower, word_tokenizer)
pruned = create_vocabulary(it)
saveRDS(pruned, "myVocab.rds")

### Test stuff ###
it_test1 = itoken(test1$review, 
                  preprocessor = prep_fun, 
                  tokenizer = tok_fun, 
                  ids = test1$new_id, 
                  progressbar = FALSE)

dtm_test1 = create_dtm(it_test1, vectorizer)


it_test2 = itoken(test2$review, 
                  preprocessor = prep_fun, 
                  tokenizer = tok_fun, 
                  ids = test2$new_id, 
                  progressbar = FALSE)

dtm_test2 = create_dtm(it_test2, vectorizer)


it_test3 = itoken(test3$review, 
                  preprocessor = prep_fun, 
                  tokenizer = tok_fun, 
                  ids = test3$new_id, 
                  progressbar = FALSE)

dtm_test3 = create_dtm(it_test3, vectorizer)

### Testing ###
set.seed(500)
NFOLDS = 10
mycv = cv.glmnet(x=dtm_train[, id], y=ytrain, 
                 family='binomial',type.measure = "auc", 
                 nfolds = NFOLDS, alpha=0)

myfit = glmnet(x=dtm_train[, id], y=ytrain, 
               lambda = mycv$lambda.min, family='binomial', alpha=0)

logit_pred1 = predict(myfit, dtm_test1[, id], type = "response")
roc_obj1 = roc(test1$sentiment, as.vector(logit_pred1))
pROC::auc(roc_obj1) 

logit_pred2 = predict(myfit, dtm_test2[, id], type = "response")
roc_obj2 = roc(test2$sentiment, as.vector(logit_pred2))
pROC::auc(roc_obj2) 

logit_pred3 = predict(myfit, dtm_test3[, id], type = "response")
roc_obj3 = roc(test3$sentiment, as.vector(logit_pred3))
pROC::auc(roc_obj3) 


##################################

