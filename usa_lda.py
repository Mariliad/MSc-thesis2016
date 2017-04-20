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
from pyspark.sql.functions import concat, col, lit, when
from pyspark.sql.functions import regexp_replace, trim, col, lower

sc = SparkContext('local','example')
sql_sc = SQLContext(sc)

df = sql_sc.read.load('./dataset/usa/*.csv', delimiter=';', format='com.databricks.spark.csv', header='true', inferSchema='true')

df = df.select('id', 'objective', 'title', 'subjects', 'foa')

# def null_as_string(x):
    # return when(col(x) != None, col(x)).otherwise(' ')

# df = (df.withColumn("objective", null_as_string("objective")).withColumn("title", null_as_string("title")).withColumn("foa", null_as_string("foa")).withColumn("subjects", null_as_string("subjects")))

# df = df.select('id', concat(col('objective'), lit(' '), col('title'), lit(' '), col('foa'), lit(' '), col('subjects')))
# df = (df.withColumnRenamed('concat(objective,  , title,  , foa,  , subjects)', 'objectives'))
df = (df.withColumnRenamed('objective', 'objectives'))

# df.count()

df = (df.withColumn('id', df.id.cast('bigint')))
df = df.na.drop(how='any')
df = (df.withColumn('id', df.id.cast('bigint')))

def removePunctuation(column):
    assert(str(type(column)) == "<class 'pyspark.sql.column.Column'>")
    columnNoPunct = regexp_replace(column, "[^a-zA-Z]", " ")
    columnLowerCase = lower(columnNoPunct)
    columnTrimmed = trim(columnLowerCase)
    return columnTrimmed


dfRemoved = df.select('id', removePunctuation(df.objectives))
dfRemoved = (dfRemoved.withColumnRenamed('trim(lower(regexp_replace(objectives, [^a-zA-Z],  )))', 'obj'))

df1 = dfRemoved.select('obj', 'id')

rdd_1 = df1.rdd.map(lambda (obj,id): Row(id = id, obj = obj.split(" ")))
dfSplit = rdd_1.toDF()

removerStopWords = StopWordsRemover(inputCol="obj", outputCol="cleanObj")
dfClean = removerStopWords.transform(dfSplit)

stopwords = ['research', 'project', 'technology', 'based', 'new', 'development', 'system', 'systems', 'develop', 'use', 'study', 
'also', 'data', 'used', 'using', 'students', 'student', 'university', 'important', 'br', 'understanding', 'program', 'provide', 'high', 
'science', 'graduate', 'undergraduate', 'field', 'well', 'two', 'work', 'proposed', 'model', 'models', 'education', 'fellowship', 
'postdoctoral', 'studies', 'materials', 'one', 'developed', 'unassigned']
remover = StopWordsRemover(inputCol="cleanObj", outputCol="cleanedObj", stopWords = stopwords)
dfCleaned = remover.transform(dfClean)

dfCleaned = dfCleaned.select('cleanedObj', 'id')

Vector = CountVectorizer(inputCol='cleanedObj', outputCol='vectors')
model = Vector.fit(dfCleaned)
result = model.transform(dfCleaned)

corpus = result.select("id", "vectors").rdd.map(lambda (x,y): [x,Vectors.fromML(y)]).cache()


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

