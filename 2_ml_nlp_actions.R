############################################################################
# 
# Machine Learning for Natural Language Processing
# 
# Javier Osorio
# University of Arizona
# 
# WICSS 2021
# 
# Classify Types of Terrorist Attacks from SATP
# 
############################################################################



############################################################################
# Roadmap
# 
# Setup
# 1. Explore the data
# 2. Split data into training and testing data
# 3. Create the document term matrix
# 4. Configure container for test data 
# 5. Configure containers for training data 
# 6. Support Vector Machine for types of terrorist actions
#    6.1 Train SVM Models
#    6.2  Predict results
# 7. Evaluate results
#    7.1 Generate confusion matrix
#    7.2 Plot Accuracy for all models
############################################################################



############################################################################
# Setup


# Clear the space
rm(list = ls())

# Load the packages
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




############################################################################
# Get the data
data <-read.csv('satp_actions.csv') # 


############################################################################
# 1. Explore the data
############################################################################

# Get variable names
names(data)

# Number of observations
dim(data)

# Show some text
head(data$news,3)

# Make sure news is character
class(data$news)

# Transform to character
data$news<-as.character(data$news)
class(data$news)

# Let's remove special characters
data %>% mutate_all(funs(gsub("[[:punct:]]", "", .)))



######################################
#  Let's explore the types of terrorist actions



# Generate data frame of stacked action categories
actions<- data %>%
  select(Kidnapping, Bombing, ArmedAssault, Other) %>%
  gather(key = "group", value = "value") %>%
  filter(value=="TRUE") %>%
  count(group, value)


# Generate graph
g.actions <- ggplot(data=actions, aes(x=group, y=n)) +
  geom_bar(stat="identity") +
  ylab("Number of stories") + xlab("Type of action")

g.actions

# Save graph
ggsave("actions.freq.pdf", width = 4, height = 3)






############################################################################
# 2. Split data into training and testing data
############################################################################

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





############################################################################
# 3. Create the document term matrix
############################################################################


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








############################################################################
# 4. Configure container for test data 
############################################################################


predN = nrow(data.test)
test.container <- create_container(dtm.test,        # Testing document term matrix
                                   labels=rep(0,predN),  # Labels
                                   testSize=1:predN,     # Number of documents to use for testing
                                   virgin=FALSE)         # FALSE = training and testing data have corresponding labels


# Remove some objects from memory to release space
#rm(dtm.test)




############################################################################
# 5. Configure containers for training data 
############################################################################


# Get the training data N. 
trainN = nrow(data.train)

# Containers for each action
names(data)

container.kid   <- create_container(dtm.train, data$Kidnapping,   trainSize=1:trainN, virgin=FALSE)           
container.bomb  <- create_container(dtm.train, data$Bombing,      trainSize=1:trainN, virgin=FALSE)           
container.armed <- create_container(dtm.train, data$ArmedAssault, trainSize=1:trainN, virgin=FALSE)           
container.other <- create_container(dtm.train, data$Other,        trainSize=1:trainN, virgin=FALSE)           




############################################################################
# 6. Support Vector Machine for types of terrorist actions
############################################################################



######################################
# 6.1 Train SVM Models


tic() 
# SVM Kidnapping
svm.kid <- train_model(container.kid, "SVM")

# SVM Bomb
svm.bomb <- train_model(container.bomb, "SVM")

# SVM Armed Assault
svm.armed <- train_model(container.armed, "SVM")

# SVM Other
svm.other <- train_model(container.other, "SVM")
toc() # Took a few secs




######################################
# 6.2  Predict results


tic() 

# SVM Kidnapping
results.kid <- classify_model(test.container, svm.kid)

# SVM Bomb
results.bomb <- classify_model(test.container, svm.bomb)

# SVM Armed Assault
results.armed <- classify_model(test.container, svm.armed)

# SVM Other
results.other <- classify_model(test.container, svm.other)

toc() # Took a few secs



############################################################################
# 7. Evaluate results
############################################################################


######################################
# 7.1 Generate confusion matrix

names(results.kid)

# Turn labels as factors testing data
data.test$Kidnapping   <-as.factor(data.test$Kidnapping)
data.test$Bombing      <-as.factor(data.test$Bombing)
data.test$ArmedAssault <-as.factor(data.test$ArmedAssault)
data.test$Other        <-as.factor(data.test$Other)



# Turn labels as factors training data
results.kid$SVM_LABEL   <-as.factor(results.kid$SVM_LABEL)
results.bomb$SVM_LABEL  <-as.factor(results.bomb$SVM_LABEL)
results.armed$SVM_LABEL <-as.factor(results.armed$SVM_LABEL)
results.other$SVM_LABEL <-as.factor(results.other$SVM_LABEL)



# Create confusion matrix
conf.mat.kid   <- confusionMatrix(results.kid$SVM_LABEL, data.test$Kidnapping)
conf.mat.bomb  <- confusionMatrix(results.bomb$SVM_LABEL, data.test$Bombing)
conf.mat.armed <- confusionMatrix(results.armed$SVM_LABEL, data.test$ArmedAssault)
conf.mat.other <- confusionMatrix(results.other$SVM_LABEL, data.test$Other)




# Print confusion matrix
conf.mat.kid
conf.mat.bomb
conf.mat.armed
conf.mat.other




######################################
# 7.2 Plot Accuracy for all models

# Extract accuracy
ac <-c(conf.mat.kid$overall[[1]],
       conf.mat.bomb$overall[[1]],
       conf.mat.armed$overall[[1]],
       conf.mat.other$overall[[1]])

# Extract lower limit
lo<-c(conf.mat.kid$overall[[3]],
      conf.mat.bomb$overall[[3]],
      conf.mat.armed$overall[[3]],
      conf.mat.other$overall[[3]])

# Extract upper limit
up<-c(conf.mat.kid$overall[[4]],
      conf.mat.bomb$overall[[4]],
      conf.mat.armed$overall[[4]],
      conf.mat.other$overall[[4]])

# Create data frame
compare <-data.frame(cbind(ac,lo,up))

compare <- compare  %>% 
  select(ac,lo,up) %>%
  mutate_if(is.factor, as.numeric)

compare$name<- c("Kidnapping", "Bombing", "Armed Assault", "Other")



# Plot accuracy across ML models
g.actions <- ggplot() + 
  geom_errorbar(data=compare, mapping=aes(x=name, ymin=up, ymax=lo), 
                width=0.1, size=1) + 
  geom_point(data=compare, mapping=aes(x=name, y=ac), 
             size=1) + 
  ylim(0,1) + ylab("Accuracy") + xlab("SVM Model") +
  coord_flip()
g.actions


# Save graph
ggsave("actions.pdf", width = 4, height = 3)






# End of script