# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages
# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats
# For example, here's several helpful packages to load in 

library(data.table)
library(Matrix)
library(xgboost)
library(caret)
library(dplyr)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# Load CSV files
cat("Read data")
df_train <- fread('../input/train.csv', sep=",", na.strings = "NA")
df_test  <- fread("../input/test.csv" , sep=",", na.strings = "NA")


data = rbind(df_train,df_test,fill=T)

features = colnames(data)

for (f in features){
  if( (class(data[[f]]) == "character") || (class(data[[f]]) == "factor"))
  {
    levels = unique(data[[f]])
    data[[f]] = factor(data[[f]], level = levels)
  }
}

# one-hot-encoding features
data = as.data.frame(data)
ohe_feats = c('X0', 'X1', 'X2', 'X3', 'X4', 'X5','X6', 'X8')
dummies = dummyVars(~ X0 + X1 + X2 + X3 + X4 + X5 + X6 + X8 , data = data)
df_all_ohe <- as.data.frame(predict(dummies, newdata = data))
df_all_combined <- cbind(data[,-c(which(colnames(data) %in% ohe_feats))],df_all_ohe)

data = as.data.table(df_all_combined)

train = data[data$ID %in% df_train$ID,]
y_train <- train[!is.na(y),y]
train = train[,y:=NULL]
train = train[,ID:=NULL]
#train_sparse <- as(data.matrix(train),"dgCMatrix")
train_sparse <- data.matrix(train)

test = data[data$ID  %in% df_test$ID,]
test_ids <- test[,ID]
test[,y:=NULL]
test[,ID:=NULL]
#test_sparse <- as(data.matrix(test),"dgCMatrix")
test_sparse <- data.matrix(test)

dtrain <- xgb.DMatrix(data=train_sparse, label=y_train)
dtest <- xgb.DMatrix(data=test_sparse);

gc()

# Params for xgboost
param <- list(booster = "gbtree",
              eval_metric = "rmse", 
              objective = "reg:linear",
              eta = .1,
              gamma = 1,
              max_depth = 4,
              min_child_weight = 1,
              subsample = .7,
              colsample_bytree = .7)


#cvFoldsList <- createFolds(1:nrow(train), k = 5)

#xgb_cv <- xgb.cv(data = dtrain,
#                 params = param,
#                 nrounds = 1500,
#                 maximize = FALSE,
#                 prediction = TRUE,
#                 folds = cvFoldsList,
#                 print.every.n = 5,
#                 early.stop.round = 50); gc()

#rounds <- which.min(xgb_cv$dt[, test.rmse.mean])
rounds = 52
mpreds = data.table(id=test_ids)

for(random.seed.num in 1:10) {
  print(paste("[", random.seed.num , "] training xgboost begin ",sep=""," : ",Sys.time()))
  set.seed(random.seed.num)
  xgb_model <- xgb.train(data = dtrain,
                         params = param,
                         watchlist = list(train = dtrain),
                         nrounds = rounds,
                         verbose = 1,
                         print.every.n = 5)
  
  vpreds = predict(xgb_model,dtest) 
  mpreds = cbind(mpreds, vpreds)    
  colnames(mpreds)[random.seed.num+1] = paste("pred_seed_", random.seed.num, sep="")
}

mpreds_2 = mpreds[, id:= NULL]
mpreds_2 = mpreds_2[, y := rowMeans(.SD)]

submission = data.table(ID=test_ids, y=mpreds_2$y)
#write.table(submission, "mercedes_xgboost.csv", sep=",", dec=".", quote=FALSE, row.names=FALSE)
