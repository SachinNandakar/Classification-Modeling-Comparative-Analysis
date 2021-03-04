#######################################
# Loading & Visualizing Iris data set #
#######################################

# NOTE: Uncomment the required line of codes to install 
#       if packages are not installed earlier.


# Load data from R datasets
data()
library(datasets)
data(iris)

# View the data
View(iris)

# Display summary statistics
head(iris, 5)
tail(iris, 5)


# summary()
summary(iris)
summary(iris$Sepal.Length)
summary(iris$Petal.Length)

# Any missing data? Check the total number of
# missing values in the iris data-set
sum(is.na(iris))


# Install package "skimr" 
# "skimr" - Compact & Flexible Summaries of data
# - expands on summary() by providing larger set of statistics
# install.packages("skimr")
library(skimr)

# Perform skim to display compact summary statistics of data
skim(iris) 

# Group data by Species & then perform skim to summarise
# using pipeline operator
iris %>% 
  dplyr::group_by(Species) %>% 
  skim() 


# Data Visualization
# R base plot()

# Panel plots
plot(iris)
plot(iris, col = "blue")

# Scatter plot
plot(iris$Sepal.Length, iris$Sepal.Width, col = "blue",
     xlab = "Sepal Length", ylab = "Sepal Width")
plot(iris$Petal.Length, iris$Petal.Width, col = "red",
     xlab = "Petal Length", ylab = "Petal Width")

# Plotting with corrplot: Correlation Analysis
# Visualising the Correlation Matrix
# install.packages("corrplot")
library(corrplot)

# Compute Correlation Matrix
CorMat <- cor(iris[,1:4]) 
head(round(CorMat,2))

# Visualise Correlation Matrix
corrplot(CorMat, method = "circle")
corrplot(CorMat, method = "number", 
         type = "upper", order = "hclust",
         tl.col = "black", tl.srt = 45,
         sig.level = 0.05, insig = "blank")

# Histogram
hist(iris$Sepal.Width, col = "blue", xlab = "Sepal Width")
hist(iris$Sepal.Length, col = "blue", xlab = "Sepal Length")
hist(iris$Petal.Width, col = "red", xlab = "Petal Width")
hist(iris$Petal.Length, col = "red", xlab = "Petal Length")

# Box-plot: Using ggplot2 package
# Uncomment the lines if packages are not installed
# install.packages("ggplot2")
# install.packages("dplyr")
library(ggplot2)
library(dplyr)

ggplot2::ggplot(iris, aes(x=iris$Sepal.Length,y= iris$Species))+
  geom_boxplot(outlier.color = "red", outlier.size = 3) + 
  coord_flip()
ggplot2::ggplot(iris, aes(x=iris$Sepal.Width,y= iris$Species))+
  geom_boxplot(outlier.color = "red", outlier.size = 3) + 
  coord_flip()
ggplot2::ggplot(iris, aes(x=iris$Petal.Length,y= iris$Species))+
  geom_boxplot(outlier.color = "red", outlier.size = 3) + 
  coord_flip()
ggplot2::ggplot(iris, aes(x=iris$Petal.Width,y= iris$Species))+
  geom_boxplot(outlier.color = "red", outlier.size = 3) + 
  coord_flip()


# Feature plots: 
# Integrating all features in single graph
# Box-plot: Features in X-axis, Factor(Species) in Y-axis
# install.packages("caret")
library(caret)
featurePlot(x = iris[,1:4], 
            y = iris$Species, 
            plot = "box",
            strip=strip.custom(par.strip.text=list(cex=.7)),
            scales = list(x = list(rot=90),
                          y = list(relation="free")),
            layout = c(4,1), 
            auto.key = list(columns = 2))

