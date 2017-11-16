install.packages("LDAvis")
install.packages("tm")
install.packages('servr') 

library(LDAvis) 
library(lda)
library(tm)
library(servr)

setwd("/home/spark/Documents/LDA_Initial_test_documents/texts/")

# read in some stopwords:
stop_words <- stopwords("SMART")
stop_words <- c(stopwords("SMART"), stopwords("english"), "0","1", "2","3","4","5","6","7","8","9","10")
stop_words 

#import data (I imported 'big.txt' using the Rtudio menu 'import dataset' and assign it to 'data' variable like below.)
filenames <- list.files(getwd(), pattern="*.txt")
filenames
files <- lapply(filenames, readLines)
getwd()

docs <- Corpus(VectorSource(files))
data <- docs
writeLines(as.character(docs[[46]]))

#remove potentially problematic symbols
toSpace <- content_transformer(function(x, pattern) { return (gsub(pattern, '', x))})
docs <- tm_map(docs, toSpace, '-')
docs <- tm_map(docs, toSpace, '’')
docs <- tm_map(docs, toSpace, '‘')
docs <- tm_map(docs, toSpace, '•')
docs <- tm_map(docs, toSpace, '”')
docs <- tm_map(docs, toSpace, '“')

#remove punctuation
docs <- tm_map(docs, removePunctuation)
#Strip digits
docs <- tm_map(docs, removeNumbers)
#remove stopwords
docs <- tm_map(docs, removeWords, stopwords("english"))
#remove whitespace
docs <- tm_map(docs, stripWhitespace)
#Good practice to check every now and then
writeLines(as.character(docs[[30]]))
#Stem document
docs <- tm_map(docs,stemDocument)


# pre-processing:
data <- gsub("'", "", data)  # remove apostrophes
data <- gsub("[[:punct:]]", " ", data)  # replace punctuation with space
data <- gsub("[[:cntrl:]]", " ", data)  # replace control characters with space
data <- gsub("^[[:space:]]+", "", data) # remove whitespace at beginning of documents
data <- gsub("[[:space:]]+$", "", data) # remove whitespace at end of documents
data <- tolower(data)  # force to lowercase

# tokenize on space and output as a list:
doc.list <- strsplit(data, "[[:space:]]+")

# compute the table of terms:
term.table <- table(unlist(doc.list))
term.table <- sort(term.table, decreasing = TRUE)

# remove terms that are stop words or occur fewer than 5 times:
del <- names(term.table) %in% stop_words | term.table < 5
term.table <- term.table[!del]
vocab <- names(term.table)


# now put the documents into the format required by the lda package:
get.terms <- function(x) {
  index <- match(x, vocab)
  index <- index[!is.na(index)]
  rbind(as.integer(index - 1), as.integer(rep(1, length(index))))
}
documents <- lapply(doc.list, get.terms)


# Compute some statistics related to the data set:
D <- length(documents)  # number of documents (2,000)
W <- length(vocab)  # number of terms in the vocab (14,568)
doc.length <- sapply(documents, function(x) sum(x[2, ]))  # number of tokens per document [312, 288, 170, 436, 291, ...]
N <- sum(doc.length)  # total number of tokens in the data (546,827)
term.frequency <- as.integer(term.table)  # frequencies of terms in the corpus [8939, 5544, 2411, 2410, 2143, ...]


# MCMC and model tuning parameters:
K <- 20
G <- 5000
alpha <- 0.02
eta <- 0.02

# Fit the model:
set.seed(357)
t1 <- Sys.time()
fit <- lda.collapsed.gibbs.sampler(documents = documents, K = K, vocab = vocab, 
                                   num.iterations = G, alpha = alpha, 
                                   eta = eta, initial = NULL, burnin = 0,
                                   compute.log.likelihood = TRUE)
t2 <- Sys.time()
t2 - t1  # about 24 minutes on laptop

theta <- t(apply(fit$document_sums + alpha, 2, function(x) x/sum(x))) 
phi <- t(apply(t(fit$topics) + eta, 2, function(x) x/sum(x)))


data2 <- list(phi = phi, 
                     theta = theta, 
                     doc.length = doc.length, 
                     vocab = vocab, 
                     term.frequency = term.frequency)

# create the JSON object to feed the visualization: 
json <- createJSON(phi = data2$phi,  
                   theta = data2$theta,  
                   doc.length = data2$doc.length,  
                   vocab = data2$vocab,  
                   term.frequency = data2$term.frequency)

serVis(json, out.dir = 'vis', open.browser = TRUE)
