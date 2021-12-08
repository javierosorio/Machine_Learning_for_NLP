############################################################################
# 
# Machine Learning for Natural Language Processing
# 
# Javier Osorio
# University of Arizona
# 
# WICSS 2021
# 
# Classify Terrorist Attacks from SATP
# 
############################################################################



############################################################################
# Roadmap
# 
# Setup
# 1. Explore the data
# 2. Split data into training and testing data
# 3. Create the document term matrix
# 4. Configure containers
# 5. Support Vector Machine model
#    5.1 Train Model
#    5.2 Predict results
#    5.3 Generate confusion matrix
# 6. Boosting
#    6.1 Train a Logit Boost Model
#    6.2 Predict  results
#    6.3 Generate confusion matrix
# 7. Bagging
#    7.1 Train a BAGGING Model
#    7.2 Predict results
#    7.3 Generate confusion matrix
# 8. Random Forest
#    8.1 Train a Random Forest Model
#    8.2 Predict results 
#    8.3 Generate confusion matrix 
# 9. Plot Accuracy for all models
#
############################################################################

### suggested boilerplate to download all missing packages automatically. 
list.of.packages <- c("dplyr", "rsample", "foreign", "ggplot2", "RTextTools", "tm", "caret", "tidymodels", "tictoc", "iml")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)


############################################################################
# Setup

# Load the packages
library(dplyr)       #MISSING
library(rsample)     #MISSING
library(foreign)     # Get data
library(ggplot2)     # Graphs
library(RTextTools)  # ML for NLP models
library(tm)          # Term document matrix
library(caret)       # Additional ML functions
library(tidymodels)  # ML fmodels
library(tictoc)      # Timer
library(iml)         # Shapley values 


# Set your working directory
setwd("E:/Dropbox/CSS UA/WICSS/ml_for_nlp")


# Get the data
data <- read.csv('satp_terrorism.csv')




######################################
# 1. Explore the data

# Get variable names
names(data)

# Number of observations
dim(data)

# Show some text
head(data$news,3)

# Make sure news is character
class(data$news)

# Transform to character
data$news <- as.character(data$news)
class(data$news)

# Let's remove special characters
data %>% mutate_all(funs(gsub("[[:punct:]]", "", .)))




######################################
# 2. Split data into training and testing data

# Set the seed for the randomization process
set.seed(123)

# Split data into training (80%) and testing (20%)
data.split <- initial_split(data, prop=0.80)

# Show the distribution of the data (Training / Test / Total)
data.split


# Generate separate databases
data.train <- training(data.split)     # Training data
data.test  <- testing(data.split)      # Test data

# Explore the data structure 
names(data.train)
names(data.test)

dim(data.train)
dim(data.test)


# Check if we have NAs
data.train %>% summarise_all(funs(sum(is.na(.))))
data.test %>% summarise_all(funs(sum(is.na(.))))





######################################
# 3. Create the document term matrix (DTM)


# Make sure news is character
class(data.train$news)
class(data.test$news)


# Create DTM for Training data
dtm.train <- create_matrix(data.train["news"],         # Identify the variable
                                language = "english",       # Set the language
                                stripWhitespace=TRUE,       # Eliminate extra space
                                minWordLength=3,            # Minimum word size
                                stemWords = TRUE,           # Stem words 
                                removeNumbers = TRUE,       # Eliminate numbers
                                removePunctuation = TRUE,   # Eliminate punctuation
                                removeStopwords = TRUE,     # Remove stop words
                                weighting=tm::weightTfIdf)  # Weights
# Explore the matrix structure
dtm.train


# Create DTM for Training data
dtm.test <- create_matrix(data.test["news"], originalMatrix=dtm.train, 
                               language = "english",
                               stripWhitespace=TRUE,
                               minWordLength=3,
                               stemWords = TRUE,
                               removeNumbers = TRUE,
                               removePunctuation = TRUE,
                               removeStopwords = TRUE,
                               weighting=tm::weightTfIdf)
# Explore the matrix structure
dtm.test



# Explore dimensions the text matrix (n, terms)
dim(dtm.train)
dim(dtm.test)



# Take a look at the terms  
text.matrix <- data.frame(as.matrix(dtm.train))         # Convert the data to a regular matrix
names(text.matrix[,c(1000:1050)])                       # Explore some terms
summary(text.matrix[,c(1000:1005)])  





######################################
# 4. Configure containers


# 4.1 Configure container for the training data
trainN = nrow(data.train)                             # Get the N of the training data

train.container <- create_container(dtm.train,        # Training document term matrix
                              data$is_relevant,       # Outcome variable
                              trainSize=1:trainN,     # Number of documents to use for training
                              virgin=FALSE)           # FALSE = training and testing data have corresponding labels



# 4.2 Configure container for the test data
predN = nrow(data.test)                                       # Get the N of the test data

test.container <- create_container(dtm.test,                  # Testing document term matrix
                                        labels=rep(0,predN),  # Labels
                                        testSize=1:predN,     # Number of documents to use for testing
                                        virgin=FALSE)         # FALSE = training and testing data have corresponding labels






############################################################################
# 5. Support Vector Machine
############################################################################


######################################
# 5.1 Train SVM Model

tic() # start timer
# Run Support vector machine
model_svm <- train_model(train.container, "SVM")
toc() # end timer   (took just a few seconds)




######################################
# 5.2 Predict outcome
results.svm <- classify_model(test.container, model_svm)

# Show first 6
results.svm[1:6,]


# Plot frequencies
ggplot(results.svm, aes(x=SVM_LABEL, y = (..count..)/sum(..count..))) + 
  geom_bar() +
  ylab("Percentage") + xlab("SVM model")
  
# Plot probabilities
ggplot(results.svm, aes(x=SVM_LABEL, y = (SVM_PROB))) + 
  geom_boxplot() + ylim(0,1) +
  ylab("Probabilities") + xlab("SVM model")





######################################
# 5.3 Generate confusion matrix

# Get column names
names(results.svm)

# First turn labels as factors
results.svm$SVM_LABEL<-as.factor(results.svm$SVM_LABEL)
data.test$is_relevant<-as.factor(data.test$is_relevant)

# Basic way
table(results.svm$SVM_LABEL,data.test$is_relevant)
conf.mat.svm <- table(results.svm$SVM_LABEL,data.test$is_relevant)
conf.mat.svm 
(conf.mat.svm[1]+conf.mat.svm[4])/sum(conf.mat.svm)  


# Another way to create a confusion matrix
conf.mat.svm <- confusionMatrix(results.svm$SVM_LABEL, data.test$is_relevant)

# Print confusion matrix
conf.mat.svm





############################################################################
# 6. Boosting
############################################################################


######################################
# 6.1 Train a Logit Boost Model  
tic()
model_boost <- train_model(train.container, "BOOSTING")
toc()  # Took about 1 minute


######################################
# 6.2 Predict  outcome
results.boost <- classify_model(test.container, model_boost)
results.boost[1:6,]


# Plot frequencies
ggplot(results.boost, aes(x=LOGITBOOST_LABEL, y = (..count..)/sum(..count..))) + 
  geom_bar() +
  ylab("Percentage") + xlab("Boosting model")


# Plot probabilities
ggplot(results.boost, aes(x=LOGITBOOST_LABEL, y = (LOGITBOOST_PROB))) + 
  geom_boxplot() + ylim(0,1) +
  ylab("Probabilities") + xlab("Boosting model")





######################################
# 6.3 Generate confusion matrix

# Turn labels as factors
results.boost$LOGITBOOST_LABEL<-as.factor(results.boost$LOGITBOOST_LABEL)
data.test$is_relevant<-as.factor(data.test$is_relevant)

# Create confusion matrix
conf.mat.boost <- confusionMatrix(results.boost$LOGITBOOST_LABEL, data.test$is_relevant)

# Print confusion matrix
conf.mat.boost




############################################################################
# 7. BAGGING
############################################################################


######################################
# ML models for NLP can easily exhaust your computer's memory
# Let's remove some objects to free memory
rm(data, 
   data.split, data.train, 
   dtm.test, dtm.train, 
   text.matrix,  
   model_boost, model_svm, 
   results.boost, results.svm)




######################################
# 7.1 Train a BAGGING Model
tic()
model_bag <- train_model(train.container, "BAGGING")
toc()  # Took about 11 minutes


######################################
# 7.2 Predict outcome
results.bag <- classify_model(test.container, model_bag)

# Show first 6
results.bag[1:6,]


# Plot frequencies
ggplot(results.bag, aes(x=BAGGING_LABEL, y = (..count..)/sum(..count..))) + 
  geom_bar() +  ylim(0,1) +
  ylab("Percentage") + xlab("Bagging model")


# Plot probabilities
ggplot(results.bag, aes(x=BAGGING_LABEL, y = (BAGGING_PROB))) + 
  geom_boxplot() + ylim(0,1) +
  ylab("Probabilities") + xlab("Bagging model")






######################################
# 7.3 Generate confusion matrix

# First turn labels as factors
results.bag$BAGGING_LABEL<-as.factor(results.bag$BAGGING_LABEL)
data.test$is_relevant<-as.factor(data.test$is_relevant)

# Create confusion matrix
conf.mat.bag <- confusionMatrix(results.bag$BAGGING_LABEL, data.test$is_relevant)

# Print accuracy
conf.mat.bag







############################################################################
# 8. Random Forest
############################################################################

# Let's remove some objects to free memory
rm(model_bag, results.bag)



######################################
# 8.1 Train a Random Forest Model
tic()
model_rf <- train_model(train.container, "RF")
toc()   # Took about 14 minutes 


######################################
# 8.2 Predict outcome Random Forest
results.rf <- classify_model(test.container, model_rf)
results.rf[1:6,]


# Plot frequencies
ggplot(results.rf, aes(x=FORESTS_LABEL, y = (..count..)/sum(..count..))) + 
  geom_bar() +  ylim(0,1) +
  ylab("Percentage") + xlab("Random Forest model")


# Plot probabilities
ggplot(results.rf, aes(x=FORESTS_LABEL, y = (FORESTS_PROB))) + 
  geom_boxplot() + ylim(0,1) +
  ylab("Probabilities") + xlab("Random Forest model")





######################################
# 8.3 Generate confusion matrix Random Forest

# First turn labels as factors
results.rf$FORESTS_LABEL<-as.factor(results.rf$FORESTS_LABEL)
data.test$is_relevant<-as.factor(data.test$is_relevant)

# Create confusion matrix
conf.mat.rf <- confusionMatrix(results.rf$FORESTS_LABEL, data.test$is_relevant)

# Print confusion matrix
conf.mat.rf







############################################################################
# 9. Plot Accuracy for all models
############################################################################

######################################
# If bagging did not run, you may want to: 
# - comment out conf.mat.bag in lines 465, 471, 477
# - comment compare$name in line 488
# - uncomment compare$name in line 489


######################################
# Extract accuracy
ac <-c(conf.mat.svm$overall[[1]],
       conf.mat.boost$overall[[1]],
       conf.mat.bag$overall[[1]],
       conf.mat.rf$overall[[1]])

# Extract lower limit
lo<-c(conf.mat.svm$overall[[3]],
      conf.mat.boost$overall[[3]],
      conf.mat.bag$overall[[3]],
      conf.mat.rf$overall[[3]])

# Extract upper limit
up<-c(conf.mat.svm$overall[[4]],
      conf.mat.boost$overall[[4]],
      conf.mat.bag$overall[[4]],
      conf.mat.rf$overall[[4]])


# Create data frame
compare <-data.frame(cbind(ac,lo,up))

compare <- compare  %>% 
  select(ac,lo,up) %>%
  mutate_if(is.factor, as.numeric)

compare$name<- c("SVM", "Boosting", "Bagging", "Random Forest" )   # Use this if you have all models
#compare$name<- c("SVM", "Boosting",            "Random Forest" )  # Use this if there is no bagging

  
######################################  
# Plot accuracy across ML models
g.relevant <- ggplot() + 
  geom_errorbar(data=compare, mapping=aes(x=name, ymin=up, ymax=lo), 
                width=0.1, size=1) + 
  geom_point(data=compare, mapping=aes(x=name, y=ac), 
             size=1) + 
  ylim(0,1) + ylab("Accuracy") + xlab("Model") +
  coord_flip()
g.relevant


# Save graph
ggsave("relevant.pdf", width = 4, height = 3)





# End of script