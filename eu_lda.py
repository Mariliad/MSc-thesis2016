sc.stop()
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.ml.feature import Tokenizer
from pyspark.ml.feature import StopWordsRemover
from pyspark.sql import Row
from pyspark.ml.feature import CountVectorizer
from pyspark.mllib.clustering import LDA, LDAModel
from pyspark.mllib.linalg import Vector, Vectors
from pyspark.sql.functions import concat, col, lit
from pyspark.sql.functions import regexp_replace, trim, col, lower

sc = SparkContext('local','example')
sql_sc = SQLContext(sc)

# Load the dataset (.csv format) as a Spark Dataframe
df = sql_sc.read.load('/dataset/cordis-fp5projects.csv', delimiter=';', format='com.databricks.spark.csv', header='true', inferSchema='true')

# Keep only the 'rcn' (unique id for each doc), the 'title' and the 'objective'
df = df.select('rcn', 'objective', 'title')

# Concatenate column 'title' with column 'objective', to obtain larger docs for LDA
df = df.select('rcn', concat(col('objective'), lit(' '), col('title')))
df = (df.withColumnRenamed('concat(objective,  , title)', 'objectives'))

# Convert 'rcn' as type 'bigint'
df = (df.withColumn('rcn', df.rcn.cast('bigint')))

# Drop rows that all their values are Null
df = df.na.drop(how='any')
df = (df.withColumn('rcn', df.rcn.cast('bigint')))


# Remove punctuation and digits. The method returns the Column with the docs (type: string), without the punctuation
def removePunctuation(column):
    assert(str(type(column)) == "<class 'pyspark.sql.column.Column'>")
    columnNoPunct = regexp_replace(column, "[^a-zA-Z]", " ")
    columnLowerCase = lower(columnNoPunct)
    columnTrimmed = trim(columnLowerCase)
    return columnTrimmed

# Change the name of the column as 'objectives'
dfRemoved = df.select('rcn', removePunctuation(df.objectives))
dfRemoved = (dfRemoved.withColumnRenamed('trim(lower(regexp_replace(objectives, [^a-zA-Z],  )))', 'obj'))

df1 = dfRemoved.select('obj', 'rcn')

# Tokenize the docs and convert the dataset to the ideal format in order to create "bag of words" later
rdd_1 = df1.rdd.map(lambda (obj,rcn): Row(rcn = rcn, obj = obj.split(" ")))
dfSplit = rdd_1.toDF()

# Remove stopwords
remover = StopWordsRemover(inputCol="obj", outputCol="cleanedObj")
dfCleaned = remover.transform(dfSplit)

dfCleaned = dfCleaned.select('cleanedObj', 'rcn')


# Get the 'bag of words' to create the vocabulary
Vector = CountVectorizer(inputCol='cleanedObj', outputCol='vectors')
model = Vector.fit(dfCleaned)
result = model.transform(dfCleaned)

# Create the corpus
corpus = result.select("rcn", "vectors").rdd.map(lambda (x,y): [x,Vectors.fromML(y)]).cache()


ldaModel = LDA.train(corpus, k=7, optimizer='online')
topics = ldaModel.topicsMatrix()
vocabArray = model.vocabulary

wordNumbers = 10  # number of words per topic
topicIndices = sc.parallelize(ldaModel.describeTopics(maxTermsPerTopic = wordNumbers))

def topic_render(topic):  # specify vector id of words to actual words
    terms = topic[0]
    result = []
    for i in range(wordNumbers):
        term = vocabArray[terms[i]]
        result.append(term)
    return result

topics_final = topicIndices.map(lambda topic: topic_render(topic)).collect()

for topic in range(len(topics_final)):
    print ("Topic" + str(topic) + ":")
    for term in topics_final[topic]:
        print (term)
    print ('\n')