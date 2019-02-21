n<-15
n
a = 12
a
24->z
z
N<-26.42
N
n
ls()
rm(n)
ls()
?ls
help(rm)
apropos("help") # "help" in name

help.search("help") # "help" in name or summary; note quotes!
help.start() # also remember the R Commands web page (link on # class page)

name<-"Mike"
name

q1<-TRUE
q1

q2<-F
q2
ls()

a <- 12+14
a

3*5

(20-4)/2

7^2

exp(2) # e^2
log(10)
log10(10)
log2(64)
pi
cos(pi)
sqrt(100)

#HW Q1

27*(38-17) #567
log(147) #4.990433
sqrt(436/12) #6.027714


a <- c(1,7, 32, 16) #Array of vectors
a

b<-1:10
b

c<-20:15
c

d <- seq(1, 5, by=0.5)
d

e<- seq(0,10, length=5)
e
f<-rep(0,5)
f

g<-rep(1:3,4)
g

h<-rep(4:6,1:3)
h

x <- rnorm(5) #rnorm(n, mean = , sd = )  Standard normal random variables
x

y <- rnorm(7, 10, 3) # Normal r.v.s with mu = 10, sigma = 3
y

z <- runif(10) # Uniform(0, 1) random variables
z

c(1, 2, 3) + c(4, 5, 6)

c(1, 2, 3, 4) + c(10, 20)

c(1, 2, 3) + c(10, 20)

sqrt(c(100, 225, 400))

d
d[3]
d[5:7]

d>2.8
d[d>2.8]

length(d)

length(d[d > 2.8])

#a = (5, 10, 15, 20, ..., 160)
a<-seq(5,160,by=5)
a
#b = (87, 86, 85, ..., 56)
b<-c(87:56)
b

#Use vector arithmetic to multiply these vectors and call the result d. 
#Select subsets of d to identify the following.
d<-a*b
d
#(a) What are the 19th, 20th, and 21st elements of d?
d[19:21]
#  (b) What are all of the elements of d which are less than 2000?
d[d<2000]
#  (c) How many elements of d are greater than 6000?
length(d[d>6000])


1:4

sum(1:4)

prod(1:4)

max(1:10)

min(1:10)

range(1:10)

X <- rnorm(10)
X

mean(X)

sort(X)

median(X)

var(X)

sd(X)

#q3 answers
sum(d)
mean(d)
median(d)
sd(d)

data(iris)
head(iris)
hist(iris$Petal.Length)
hist(iris[,4])# alternative specification

boxplot(iris$Petal.Length)
boxplot(Petal.Length~Species, data=iris) # Formula description,
# side-by-side boxplots

#Q4 

data(cars)
head(cars)
#(a) Plot a histogram of distance using the hist function
hist(cars$dist,main = "Histogram of Distance", xlab = "Distance", ylab = "Frequency / Count", col ="blue")

ggplot(data = cars,aes(cars$dist))+ geom_histogram(breaks=seq(0, 120, by = 20))

#(b) Generate a boxplot of speed.
boxplot(cars$speed, col = "green", main="Boxplot of Speed", xlab = "Speed", ylab = "Speed value range")

#(c) Use the plot(,) function (e.g. plot(variableX, variableY) to create a 
#scatterplot of dist against speed.
plot(x = cars$dist, y = cars$speed, main = "Scatterplot of Distance against Speed", xlab = "Distance", ylab = "Speed")

ggplot(data = cars, aes(dist, speed))+ geom_point()

#[Note: You can also create the graphs in parts (a) to (c) using ggplot2 package].

#Part II

library(mlbench)
data(Glass)
str(Glass)
?Glass

library(ggplot2)
#(a) Using visualizations, explore the predictor variables to understand their 
#distributions as well as the relationships between predictors. 
#Provide the pairwise scatter plots and investigate the correlation matrix.

#To explore predictor variables we can use either histograms or density plots. I have showed histogram 
#an density in one plot for each of the glass types. 

#histogram of the target variable i.e. Y
ggplot(Glass,aes(x=Type))+geom_bar()+ggtitle("Observation count by Type of Glass")
#The above command tells me that in my Glass dataset most of the glass types
#are of type 1 or of type 2.

#Histogram of all the predictor variables
GlassRI<-ggplot(Glass, aes(x=RI)) + 
  geom_histogram(color="black", fill="green")+ggtitle("Histogram of RI")
GlassNa<-ggplot(Glass, aes(x=Na)) + 
  geom_histogram(color="black", fill="green")+ggtitle("Histogram of Na")
GlassMg<-ggplot(Glass, aes(x=Mg)) + 
  geom_histogram(color="black", fill="green")+ggtitle("Histogram of Mg")
GlassAl<-ggplot(Glass, aes(x=Al)) + 
  geom_histogram(color="black", fill="green")+ggtitle("Histogram of Al")

grid.arrange(GlassRI, GlassNa,GlassMg,GlassAl, ncol=2)

GlassSi<-ggplot(Glass, aes(x=Si)) + 
  geom_histogram(color="black", fill="green")+ggtitle("Histogram of Si")
GlassK<-ggplot(Glass, aes(x=K)) + 
  geom_histogram(color="black", fill="green")+ggtitle("Histogram of K")
GlassCa<-ggplot(Glass, aes(x=Ca)) + 
  geom_histogram(color="black", fill="green")+ggtitle("Histogram of Ca")
GlassBa<-ggplot(Glass, aes(x=Ba)) + 
  geom_histogram(color="black", fill="green")+ggtitle("Histogram of Ba")

grid.arrange(GlassSi, GlassK, GlassCa, GlassBa, ncol=2)

GlassFe<-ggplot(Glass, aes(x=Fe)) + 
  geom_histogram(color="black", fill="green")+ggtitle("Histogram of Fe")
GlassFe


ggplot(data = cars, aes(dist, speed))+ geom_point()

##Correlation Matrix
library(corrplot)
Glasscorr <- Glass[,1:length(Glass)]
Glasscorr<-data.matrix(Glasscorr)
round(cor(Glasscorr),2)
corrplot.mixed(cor(Glasscorr),lower = "number",upper = "circle")
#corrplot(cor(Glasscorr), method = "circle")

#Correlation value of each predictor w.r.t. Type
cor(Glass[,-10],as.numeric(Glass[,10]))

#pairwise scatterplots of all the attributes
pairs(Type~.,data=Glass, main="Simple Scatterplot Matrix")
pairs(Glass[,-10],main="Scatterplot Matrix for Glass Dataset")



#(b) Do there appear to be any outliers in the data? Are any predictors skewed?

#Exploring the predictor variables w.r.t. outliers(boxplot) and distributions(histogram)
#Boxplots ---- for Outlier Analysis
boxplot(Glass)

BoxRI<-boxplot(Glass$RI, col = "green", main="Boxplot of RI", xlab = "RI", ylab = "Values")
BoxNa<-boxplot(Glass$Na, col = "green", main="Boxplot of Na", xlab = "Na", ylab = "Values")
BoxMg<-boxplot(Glass$Mg, col = "green", main="Boxplot of Mg", xlab = "Mg", ylab = "Values")
BoxAl<-boxplot(Glass$Al, col = "green", main="Boxplot of Al", xlab = "Al", ylab = "Values")
boxplot(Glass[,1:3], col = "green")

grid()
grid.arrange(BoxRI, BoxNa, BoxMg, BoxAl, ncol=2)

BoxSi<-boxplot(Glass$Si, col = "green", main="Boxplot of Si", xlab = "Si", ylab = "Values")
BoxK<-boxplot(Glass$K, col = "green", main="Boxplot of K", xlab = "K", ylab = "Values")
BoxCa<-boxplot(Glass$Ca, col = "green", main="Boxplot of Ca", xlab = "Ca", ylab = "Values")
BoxBa<-boxplot(Glass$Ba, col = "green", main="Boxplot of Ba", xlab = "Ba", ylab = "Values")
BoxFe<-boxplot(Glass$Fe, col = "green", main="Boxplot of Fe", xlab = "Fe", ylab = "Values")


grid.arrange(BoxSi, BoxK, BoxCa, BoxBa, ncol=2)

plot.BoxRI <- ggplot(Glass, aes(x = Type, y = RI)) +geom_boxplot() + ggtitle("Boxplot of RI per Type")
plot.BoxRI

plot.BoxNa <- ggplot(Glass, aes(x = Type, y = Na)) +geom_boxplot() + ggtitle("Boxplot of Na per Type")
plot.BoxNa

plot.BoxMg <- ggplot(Glass, aes(x = Type, y = Mg)) +geom_boxplot() + ggtitle("Boxplot of Mg per Type")
plot.BoxMg

plot.BoxAl <- ggplot(Glass, aes(x = Type, y = Al)) +geom_boxplot() + ggtitle("Boxplot of Al per Type")
plot.BoxAl

plot.BoxSi <- ggplot(Glass, aes(x = Type, y = Si)) +geom_boxplot() + ggtitle("Boxplot of Si per Type")
plot.BoxSi

plot.BoxK <- ggplot(Glass, aes(x = Type, y = K)) +geom_boxplot() + ggtitle("Boxplot of K per Type")
plot.BoxK

plot.BoxCa <- ggplot(Glass, aes(x = Type, y = Ca)) +geom_boxplot() + ggtitle("Boxplot of Ca per Type")
plot.BoxCa

plot.BoxBa <- ggplot(Glass, aes(x = Type, y = Ba)) +geom_boxplot() + ggtitle("Boxplot of Ba per Type")
plot.BoxBa

plot.BoxFe <- ggplot(Glass, aes(x = Type, y = Fe)) +geom_boxplot() + ggtitle("Boxplot of Fe per Type")
plot.BoxFe

#Compute skewness

library(e1071)
skewness(Glass$RI)
skewness(Glass$Na)
skewness(Glass$Mg)
skewness(Glass$Al)
skewness(Glass$Si)
skewness(Glass$K)
skewness(Glass$Ca)
skewness(Glass$Ba)
skewness(Glass$Fe)

GlassData<-Glass[,-10]

skewValues<- apply(GlassData,2,skewness)
skewValues

#Histograms with density plot for checking skewness

ggplot(Glass, aes(x=RI)) + geom_histogram(aes(y=..density..),colour="black", fill="white") +
  geom_density(alpha=.5, colour="red") 

ggplot(Glass, aes(x=Na)) + geom_histogram(aes(y=..density..),colour="black", fill="white") +
  geom_density(alpha=.5, colour="red") 

ggplot(Glass, aes(x=Mg)) + geom_histogram(aes(y=..density..),colour="black", fill="white") +
  geom_density(alpha=.5, colour="red") 

ggplot(Glass, aes(x=Al)) + geom_histogram(aes(y=..density..),colour="black", fill="white") +
  geom_density(alpha=.5, colour="red") 

ggplot(Glass, aes(x=Si)) + geom_histogram(aes(y=..density..),colour="black", fill="white") +
  geom_density(alpha=.5, colour="red") 

ggplot(Glass, aes(x=K)) + geom_histogram(aes(y=..density..),colour="black", fill="white") +
  geom_density(alpha=.5, colour="red") 

ggplot(Glass, aes(x=Ca)) + geom_histogram(aes(y=..density..),colour="black", fill="white") +
  geom_density(alpha=.5, colour="red") 

ggplot(Glass, aes(x=Ba)) + geom_histogram(aes(y=..density..),colour="black", fill="white") +
  geom_density(alpha=.5, colour="red") 

ggplot(Glass, aes(x=Fe)) + geom_histogram(aes(y=..density..),colour="black", fill="white") +
  geom_density(alpha=.5, colour="red") 
#Observations:  1. K and Mg has second modes near zero
#               2. Ca, Ba, Fe, RI has skewness


#(c) Are there any relevant transformations of one or more predictors that might
#improve the classification model? (Hint: You could transform the predictors using 
#the BoxCox Transformation. This can be done using mathematical formulation, or 
#using the "preprocess" function in the AppliedPredictiveModeling package).
library(MASS)
library(forecast)

lambdaMg = BoxCox.lambda(Glass$Mg)
lambdaMg

library(caret)

Glasstrans<-preProcess(Glass[,-10], method = "BoxCox")
transformed<-predict(Glasstrans, Glass[,-10])
skewValuesAfterTrans<-apply(transformed[,-10],2,skewness)

skewValues
skewValuesAfterTrans

##Ba, Fe, Mg, K

####
CoxRI<-BoxCoxTrans(Glass$K)
CoxRI

#part III

library(mlbench)
data(Soybean)
##See ?Soybean for details
?Soybean
class(Soybean)
str(Soybean)

#(a) Investigate the frequency distributions for the categorical predictors.
#Are there any extremely unbalanced categorical predictors? In the extreme case, 
#if there is only one value for the predictor, it is called a degenerate case. 
#Such degenerate predictors should be removed from subsequent analysis.

library(dplyr)
summary(Soybean)

#Frequency Distributions for the categorical predictors

par(mfrow=c(3,3), mai = c(1, 0.1, 0.1, 0.1)) 
for(i in 1:9)
{  hist(as.numeric(Soybean[,i]), main = colnames(Soybean[i]), col = "yellow", xlab = colnames(Soybean[i]))}
  
par(mfrow=c(3,3), mai = c(1, 0.1, 0.1, 0.1)) 
for(i in 10:18)
{  hist(as.numeric(Soybean[,i]), main = colnames(Soybean[i]), col = "red", xlab = colnames(Soybean[i]))}

par(mfrow=c(3,3), mai = c(1, 0.1, 0.1, 0.1)) 
for(i in 19:27)
{  hist(as.numeric(Soybean[,i]), main = colnames(Soybean[i]), col = "blue", xlab = colnames(Soybean[i]))}

par(mfrow=c(3,3), mai = c(1, 0.1, 0.1, 0.1)) 
for(i in 28:36)
{  hist(as.numeric(Soybean[,i]), main = colnames(Soybean[i]), col = "green", xlab = colnames(Soybean[i]))}


summary(Soybean)

table(Soybean$Class)
barplot(table(Soybean$Class))

table(Soybean$date)
barplot(table(Soybean$date))

table(Soybean$plant.stand)
barplot(table(Soybean$plant.stand))

table(Soybean$precip)
barplot(table(Soybean$precip))

table(Soybean$temp)
barplot(table(Soybean$temp))

table(Soybean$hail)
barplot(table(Soybean$hail))

table(Soybean$crop.hist)
barplot(table(Soybean$crop.hist))

table(Soybean$area.dam)
barplot(table(Soybean$area.dam))

table(Soybean$sever)
barplot(table(Soybean$sever))

table(Soybean$seed.tmt)
barplot(table(Soybean$seed.tmt))

table(Soybean$germ)
barplot(table(Soybean$germ))

table(Soybean$plant.growth)
barplot(table(Soybean$plant.growth))

table(Soybean$leaves)
barplot(table(Soybean$leaves))

table(Soybean$leaf.halo)
barplot(table(Soybean$leaf.halo))

table(Soybean$leaf.marg)
barplot(table(Soybean$leaf.marg))

table(Soybean$leaf.size)
barplot(table(Soybean$leaf.size))

table(Soybean$leaf.shread)
barplot(table(Soybean$leaf.shread))

table(Soybean$leaf.malf)
barplot(table(Soybean$leaf.malf))

table(Soybean$leaf.mild)
barplot(table(Soybean$leaf.mild))

table(Soybean$stem)
barplot(table(Soybean$stem))

table(Soybean$lodging)
barplot(table(Soybean$lodging))

table(Soybean$stem.cankers)
barplot(table(Soybean$stem.cankers))

table(Soybean$canker.lesion)
barplot(table(Soybean$canker.lesion))

table(Soybean$fruiting.bodies)
barplot(table(Soybean$fruiting.bodies))

table(Soybean$ext.decay)
barplot(table(Soybean$ext.decay))

table(Soybean$mycelium)
barplot(table(Soybean$mycelium))

table(Soybean$int.discolor)
barplot(table(Soybean$int.discolor))

table(Soybean$sclerotia)
barplot(table(Soybean$sclerotia))

table(Soybean$fruit.pods)
barplot(table(Soybean$fruit.pods))

table(Soybean$fruit.spots)
barplot(table(Soybean$fruit.spots))

table(Soybean$seed)
barplot(table(Soybean$seed))

table(Soybean$mold.growth)
barplot(table(Soybean$mold.growth))

table(Soybean$seed.discolor)
barplot(table(Soybean$seed.discolor))

table(Soybean$seed.size)
barplot(table(Soybean$seed.size))

table(Soybean$shriveling)
barplot(table(Soybean$shriveling))

table(Soybean$roots)
barplot(table(Soybean$roots))

#To identify and remove NearZero Variance Predictors
library(caret)
zerocol = nearZeroVar(Soybean)
colnames( Soybean )[zerocol]
Soybean = Soybean[,-zerocol] 

#(b) Roughly 18% of the data are missing. Are there particular predictors that are more
#likely to be missing? Is the pattern of missing data related to the classes?

#Counting NA's in each column:

apply(Soybean[,2:33],2,function(x){sum(is.na(x))})

# To check which particular predictor have more NA's then others:

Soybean$napresent = apply(Soybean[,2:33],1,function(x){sum(is.na(x))>0})
table(Soybean[,c(1,34)])

#(c) Develop a strategy for handling missing data, either by eliminating predictors
#or imputation.

# For imputation of data for the NA's
 
library(caret)
#preProcess(Soybean[,2:33],method="knnImpute",na.remove=FALSE) 

summary(Soybean)

#Imputation using Hmisc
library(mice)
str(Soybean)

Soybean$plant.stand <- as.factor(as.numeric(Soybean$plant.stand))
Soybean$precip <- as.factor(as.numeric(Soybean$precip))
Soybean$temp <- as.factor(as.numeric(Soybean$temp))
Soybean$germ <- as.factor(as.numeric(Soybean$germ))
Soybean$leaf.size <- as.factor(as.numeric(Soybean$leaf.size))

str(Soybean)

#Imputing the missing values
install.packages("Hmisc")
library(Hmisc)

for(i in 1:ncol(Soybean)){
  Soybean[,i]=impute(as.factor(Soybean[,i]),mode)
}

sum(is.na(Soybean))

# Soybeannew = sapply(Soybean,function(x){impute(x,mode)})
# sum(is.na(Soybeannew))
