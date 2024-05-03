# Load the required library
library(e1071)

# Read the data from the file
data <- read.table("2020_bn_nb_data.txt", header = TRUE)

# Convert character variables into factors
data[] <- lapply(data, as.factor)

# Define the variables for prediction
EC100 <- "DD"
IT101 <- "CC"
MA101 <- "CD"

# Train the Naive Bayes classifier
nb_classifier <- naiveBayes(PH100 ~ EC100 + IT101 + MA101, data = data)

# Create new data frame for prediction
new_data <- data.frame(EC100 = factor(EC100, levels = levels(data$EC100)),
                       IT101 = factor(IT101, levels = levels(data$IT101)),
                       MA101 = factor(MA101, levels = levels(data$MA101)))

# Predict the grade in PH100
prediction <- predict(nb_classifier, newdata = new_data)

# Print the predicted grade
print(prediction)

prediction via naive bias






# Load the required library
library(e1071)

# Read the data from the file
data <- read.table("2020_bn_nb_data.txt", header = TRUE)

# Convert character variables into factors
data[] <- lapply(data, as.factor)

# Define the variables for prediction
EC100 <- "DD"
IT101 <- "CC"
MA101 <- "CD"

# Train the Naive Bayes classifier
nb_classifier <- naiveBayes(PH100 ~ EC100 + IT101 + MA101, data = data)

# Create new data frame for prediction
new_data <- data.frame(EC100 = factor(EC100, levels = levels(data$EC100)),
                       IT101 = factor(IT101, levels = levels(data$IT101)),
                       MA101 = factor(MA101, levels = levels(data$MA101)))

# Predict the grade in PH100
prediction <- predict(nb_classifier, newdata = new_data)

# Print the predicted grade
print(prediction)

prediction via naive bias