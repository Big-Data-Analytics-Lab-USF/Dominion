setwd("C:/Users/13212/OneDrive/Documents/College/USF/GA Position/Dominon/raw NER")

library(readxl)
library(dplyr)
library(sqldf)
library(ggplot2)
library(tm)
library(lsa)



#load in data
NER_dates_halfOne <- read.csv("C:/Users/13212/OneDrive/Documents/College/USF/GA Position/Dominon/raw NER/Larger Model/completed/NER_complete_Firsthalf.csv")
Ner_dates_halfTwo <- read.csv("C:/Users/13212/OneDrive/Documents/College/USF/GA Position/Dominon/raw NER/Larger Model/completed/ner_output_8723 - end.csv")
Combined <- read.csv("C:/Users/13212/OneDrive/Documents/College/USF/GA Position/Dominon/combined.csv")
#correct tweet id count
Ner_dates_halfTwo$tweet_id <- Ner_dates_halfTwo$tweet_id + 8670
NER_dates <- rbind(NER_dates_halfOne,Ner_dates_halfTwo)
#clean incorrect dominion tagging
NER_dates <- NER_dates %>%
  mutate(tag = ifelse(grepl("Dominion", entity) & tag == "GPE", "ORG", tag))
#clean locations
NER_dates$entity <- gsub("(?i)PENNSYLVANIA|State Pennsylvaniya", "Pennsylvania", NER_dates$entity, ignore.case = TRUE)
NER_dates$entity <- gsub("china|communist china|	China China", "China",NER_dates$entity, ignore.case = TRUE)
NER_dates$entity <- gsub("Georgia","Georgia",NER_dates$entity, ignore.case = TRUE)
NER_dates$entity <- gsub("Texas|state texas", "Texas",NER_dates$entity, ignore.case = TRUE)
NER_dates$entity <- gsub("Antrim|Michigan Antrim|Antrim County","Antrim County",NER_dates$entity, ignore.case = TRUE)
NER_dates$entity <- gsub("Michigan","Michigan",NER_dates$entity, ignore.case = TRUE)
NER_dates <- NER_dates %>%
  mutate(entity = case_when(
    entity %in% c("EEUU", "US", "U.S", "EE.UU", "USA", "U.S.", "America") ~ "United States",
    TRUE ~ entity
  ))
NER_dates <- NER_dates %>%
  mutate(entity = case_when(
    entity %in% c("GA") ~ "Georgia",
    TRUE ~ entity
  ))
NER_dates <- NER_dates %>%
  mutate(entity = case_when(
    entity %in% c("PA") ~ "Pennsylvania",
    TRUE ~ entity
  ))
#Get counts of unique locations for entire dataset  
locationCounts <- sqldf("SELECT entity, COUNT(entity) FROM NER_dates WHERE tag ='GPE' 
                        GROUP BY entity ORDER BY COUNT(entity) DESC")




#add dates 
#change query id 
Combined$Query.Id <- 0:(nrow(Combined) - 1)

# Use match to find matching indices
matching_indices <- match(NER_dates$tweet_id, Combined$Query.Id)

# Create a new date column based on the matches
NER_dates$date <- Combined$Date[matching_indices]


#create date frequency plot
#Convert dates
NER_dates$date <- as.Date(NER_dates$date, format = "%Y-%m-%d")

#subset location
subsetloc <- NER_dates[NER_dates$tag == 'GPE', ]


# Count occurrences of each entity on each date using SQL
result <- sqldf("SELECT entity, date, COUNT(entity) AS count
                 FROM subsetloc
                 GROUP BY entity, date
                 ORDER BY count DESC, entity")


#bar graph
top_10_entities <- result %>%
  group_by(entity) %>%
  summarize(count = sum(count)) %>%
  top_n(10, wt = count) %>%
  arrange(desc(count))


#bar chart with only top 10
# Filter the data to select only the first 10 unique entities
top_entities <- unique(result$entity)[1:10]
result_filtered <- result[result$entity %in% top_entities, ]

# Create the ggplot with the filtered data
ggplot(result_filtered, aes(x = date, y = count, fill = as.factor(entity))) +
  geom_bar(stat = "identity") +
  labs(
    x = "Date",
    y = "Count",
    title = "Location counts with duplicates",
    fill = "Entity"
  ) +
  scale_fill_discrete(name = "Entity") +
  scale_x_date(date_labels = "%Y-%m-%d", date_breaks = "1 day", expand = c(0, 0)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) 

         
#cosine similarity
#closer to one = more similar
#closer to 0 = more different 
#######################################################################################
#finding cosine simularity for rows one and two, need to apply to entire dataset
#have to run with cleaned tweets 
cs_clean_tweets <- read.csv("C:/Users/13212/OneDrive/Documents/College/USF/GA Position/Dominon/results/november/cleaned_tweets.csv")

#clean empty rows
clean <- cs_clean_tweets[cs_clean_tweets$cleaned != "", ]

clean <- clean[1:500, ]
#between 200-300 is causing issue?
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


subset <- read.csv("C:/Users/13212/OneDrive/Documents/College/USF/GA Position/Dominon/raw NER/Larger Model/Results/LargeNERcosine.csv")

subsetfilered <- subset[subset$similarities < 0.8, ]
#13078
subsetfilered05 <- subset[subset$similarities < 0.5, ]
#12891
#only looking at 2 lines to find duplicates need to get fid off all duplicates

subset <- read.csv("C:/Users/13212/OneDrive/Documents/College/USF/GA Position/Dominon/raw NER/Larger Model/Results/LargeNERcosine.csv")
#another way to remove duplicates
alt <- subsetfilered[!duplicated(subsetfilered$cleaned), ]


############################################################################

#new graph cosine duplicates removed


colnames(alt)[colnames(alt) == "index"] <- "tweet_id"

altgraph <- inner_join(subsetloc, alt, by = "tweet_id")

# Select only the columns from the first dataset
altgraph <- select(altgraph, everything())

altnodup <- sqldf("SELECT entity, date, COUNT(entity) AS count
                 FROM altgraph
                 GROUP BY entity, date
                 ORDER BY COUNT(entity) DESC, entity")

#write.csv(altnodup, 'C:/Users/13212/OneDrive/Documents/College/USF/GA Position/Dominon/raw NER/Larger Model/Results/LOCcountwithoutdup.csv')

#bar graph
top_10_entities_alt <- altnodup %>%
  group_by(entity) %>%
  summarize(count = sum(count)) %>%
  top_n(10, wt = count) %>%
  arrange(desc(count))


#bar chart with only top 10
# Filter the data to select only the first 10 unique entities
top_entities_alt <- unique(altnodup$entity)[1:10]
result_filtered_alt <- altnodup[altnodup$entity %in% top_entities_alt, ]

altnodup$date <- as.Date(altnodup$date)

ggplot(result_filtered_alt, aes(x = date, y = count, fill = as.factor(entity))) +
  geom_bar(stat = "identity") +
  labs(
    x = "Date",
    y = "Count",
    title = "Location counts without duplicates",
    fill = "Entity"
  ) +
  scale_fill_discrete(name = "Entity") +
  scale_x_date(date_labels = "%Y-%m-%d", date_breaks = "1 day", expand = c(0, 0)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) 



