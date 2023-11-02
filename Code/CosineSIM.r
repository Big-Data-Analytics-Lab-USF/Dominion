#load in clean tweets
cs_clean_tweets <- read.csv("C:/Users/13212/OneDrive/Documents/College/USF/GA Position/Dominon/results/november/cleaned_tweets.csv")

#clean empty rows
clean <- cs_clean_tweets[cs_clean_tweets$cleaned != "", ]

compare_strings_cosine_dataset <- function(dataset) {
  n_rows <- nrow(dataset)
  similarities <- numeric(n_rows)
  
  for (i in 1:(n_rows - 1)) {
    one <- dataset[i, 3]
    two <- dataset[i + 1, 3]
    
    # Split the strings into vectors of words
    word_vector1 <- unlist(strsplit(one, " "))
    word_vector2 <- unlist(strsplit(two, " "))
    
    # Create temporary directory and write strings to files
    td <- tempfile()
    dir.create(td)
    write(c(one), file = paste(td, "D1", sep = "/"))
    write(c(two), file = paste(td, "D2", sep = "/"))
    
    # Read files into a document-term matrix
    myMatrix <- textmatrix(td, minWordLength = 1)
    
    # Calculate cosine similarity
    res <- lsa::cosine(myMatrix[, 1], myMatrix[, 2])
    
    # Store the result in the similarities vector
    similarities[i] <- res
    
    # Clean up temporary files and directory
    unlink(td, recursive = TRUE)
  }
  
  return(similarities)
}
similarities <- compare_strings_cosine_dataset(clean)
similarities <- as.data.frame(similarities)

subset <- cbind(clean, similarities)

#saving as csv since its takes a long time to run
#write.csv(subset, 'C:/Users/13212/OneDrive/Documents/College/USF/GA Position/Dominon/raw NER/Larger Model/LargeNERcosine.csv')
#subset <- read.csv("C:/Users/13212/OneDrive/Documents/College/USF/GA Position/Dominon/raw NER/Larger Model/Results/LargeNERcosine.csv")

#deleting scores over certain similarity score 
subsetfilered <- subset[subset$similarities < 0.8, ]
subsetfilered05 <- subset[subset$similarities < 0.5, ]
