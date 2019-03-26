from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, ArrayType, StringType, IntegerType, FloatType
from pyspark.ml.feature import NGram
from pyspark.sql.functions import split, explode, monotonically_increasing_id, udf, col, least, greatest
from pyspark.sql.functions import sum as suma

from math import log


def decimal(y):
    if y == 'A':
        return 0b00
    if y == 'C':
        return 0b01
    if y == 'G':
        return 0b10
    return 0b11


def binary(x):
    return map(lambda y: decimal(y), x)


def reducer_concat(l):
    r = 0
    for x in l.split(" "):
        r = (r << 2) + decimal(x)
    return r


idx = lambda p, s: ((~(p ^ (0b11 << (2 * s))) & (0b11 << (2 * s))) >> (2 * s))


def spaced_words(w0, w1, m):
    if ~(w0 ^ w1) & m == m:
        return w0 & m
    else:
        return None


def udf_spaced_words(mask):
    return udf(lambda x, y: spaced_words(x, y, mask), IntegerType())


def calculate_score(w0, w1, m, k, S):
    score = 0

    # Calculate the don't care mask.
    dont_care_mask = (m ^ ((0b1 << (2 * k)) - 1))

    for i in range(k):
        if idx(dont_care_mask, i) == 0b11:
            score += (S[idx(w0, i)][idx(w1, i)])

    return score


def udf_score(mask, k, Score):
    return udf(lambda x, y: calculate_score(x, y, mask, k, Score), IntegerType())


def jukes_cantor(w0, w1, m, k):
    score = 0

    # Calculate the don't care mask.
    dont_care_mask = (m ^ ((0b1 << (2 * k)) - 1))

    for i in range(k):
        if idx(dont_care_mask, i) == 0b11:
            if not idx(w0, i) == idx(w1, i):
                score += 1

    return score


def udf_jukes_cantor(mask, k):
    return udf(lambda x, y: jukes_cantor(x, y, mask, k), IntegerType())


JukesCantor = lambda p: -0.75*log(1-1.25*p)


def random_pattern(l, w):
    from numpy.random import choice

    idx, n = choice(l - 1, w), "11"

    for a in range(l - 1):
        if a in idx:
            n += "11"
        else:
            n += "00"

    return int(n, 2)


def run(w=14, l=114, threshold=0):
    # Creating Data Frame and filtering by threshold
    pattern, k = random_pattern(l, w), w

    Chiaromonte = [
        [91, -114, -31, -123],
        [-114, 100, -125, -31],
        [-31, -125, 100, -114],
        [-123, -31, -114, 91]
    ]

    spark = SparkSession.builder.appName('Distributed FSWM').getOrCreate()

    df = spark.read.text("data/example.fasta")

    # Read the sequences
    sequences = df.where(~df.value.contains('>')).rdd.map(list).map(lambda x: (x[0].encode('ascii'))).map(list)

    # Defining schema for data frame
    schema = StructType([StructField("id", IntegerType()), StructField("Sequence", ArrayType(StringType()))])

    df = spark.createDataFrame(
        (tuple([_id, data[0]]) for _id, data in enumerate(map(lambda x: [x], sequences.take(2)))),
        schema=schema
    )

    # Creating ngrams
    ngram = NGram(n=w, inputCol="Sequence", outputCol="ngrams")
    df_clean = ngram.transform(df).select(["id", "ngrams"])

    # Exploding ngrams into the data frame
    df_explode = df_clean.withColumn('ngrams', explode('ngrams'))

    # Defining the reducer
    # Create your UDF object (which accepts your python function called "my_udf")
    udf_object = udf(lambda y: reducer_concat(y), IntegerType())


    # Here we should have for all the sequences

    df_w0 = df_explode.where(df_clean.id == 0)
    df_w0 = df_w0.withColumn("id0", monotonically_increasing_id() + 1).withColumnRenamed('ngrams', 'w0').select('id0','w0')
    df0 = df_w0.withColumn("word0", udf_object(df_w0.w0)).select("id0", "word0")
    df0.show()

    df_w1 = df_explode.where(df_clean.id == 1)
    df_w1 = df_w1.withColumn("id1", monotonically_increasing_id() + 1).withColumnRenamed('ngrams', 'w1').select('id1','w1')
    df1 = df_w1.withColumn("word1", udf_object(df_w1.w1)).select("id1", "word1")
    df1.show(truncate=False)

    df_result = df0.crossJoin(df1) \
        .withColumn("spaced_word", udf_spaced_words(pattern)(col("word0"), col("word1"))) \
        .where(col("spaced_word").isNotNull()) \
        .withColumn("score", udf_score(pattern, k, Chiaromonte)(col("word0"), col("word1"))) \
        .where(col("score") > threshold) \
        .orderBy(["spaced_word", "score"], ascending=False) \
        .withColumn("min", least(col("id0"), col("id1"))) \
        .withColumn("max", greatest(col("id0"), col("id1"))) \
        .drop_duplicates(subset=["spaced_word", "min"]) \
        .drop_duplicates(subset=["spaced_word", "max"]) \
        .withColumn("JukesCantor", udf_jukes_cantor(pattern, k)(col("word0"), col("word1")))

    df_result.show()

    p = df_result.agg(suma("JukesCantor")).collect()[0][0] * 1.0 / ((k - bin(pattern).count("1") / 2) * df_result.count())
    
    print(JukesCantor(p))


if __name__ == "__main__":
    run()
