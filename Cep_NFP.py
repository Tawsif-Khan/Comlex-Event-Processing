from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import pandas as pd
from pyspark.sql.types import *
from pyspark.ml.feature import StringIndexer

class Cep_NFP:

    modelFile = 'model.model'
    assembler = None
    login_id = ''

    def __init__(self):
        print('Connecting...')
        self.__createSparkSession('My_session')

    def createDataframe(self,login_id,zone,speed,duration,downloaded,uploaded):
        self.login_id = login_id
        schema = StructType([
                StructField("zone", StringType(), True),
                StructField("speed", StringType(), True),
                StructField('duration',IntegerType(),True),
                StructField('downloaded',IntegerType(),True),
                StructField('uploaded',IntegerType(),True),
                StructField('active_in_zone',IntegerType(),True)])
        self.data = self.spark.createDataFrame([(zone,speed,duration,downloaded,uploaded,self.__getActiveUsers(self.spark,zone))],['zone','speed','duration', 'downloaded', 'uploaded', 'active_in_zone'],schema)
        self.__setIndexed('zone','zone_index')
        self.__setIndexed('speed','speed_index')
        self.__setFeatureCols(['zone_index','speed_index','duration', 'downloaded', 'uploaded', 'active_in_zone'], 'features')
        self.__setLabelCol('fault')


    def newSessionCreated(self,zone):
        zones = self.spark.read.csv('zones.csv', inferSchema=True, header=True)
        zones = zones.withColumn(zone, zones[zone] + 1)
        zones = zones.drop('_c0')
        dd = pd.DataFrame(data=zones.head(1), columns=zones.columns)
        dd.to_csv("zones.csv")

    def getModel(self):
        try:
            self.model = RandomForestClassificationModel.load('model.model')
            return self.model
        except:
            return None

    def __createSparkSession(self, sessionName):
        print(bcolors.ENDC,bcolors.OKBLUE, 'Creating session..', bcolors.ENDC)
        self.spark = SparkSession.builder.appName(sessionName).getOrCreate() #.conf(SparkConf().('local[0'))
        return self.spark

    def openCSVFile(self, path):
        print(bcolors.ENDC,bcolors.OKBLUE, 'Loading data...',bcolors.ENDC)
        self.data = self.spark.read.csv(path, inferSchema=True, header=True)
        self.__setFeatureCols(['zone_index', 'speed_index', 'duration', 'downloaded', 'uploaded', 'active_in_zone'],
                            'features')
        self.__setLabelCol('fault')

    def getSplitedData(self, train, test):
        return (self.data.randomSplit([train, test]))

    def initRandomForest(self):
        self.rf = RandomForestClassifier(labelCol=self.labelCol, featuresCol=self.featureCol)


    def trainModel(self,trainSet):
        self.model = self.rf.fit(trainSet)
        self.__saveModel(self.model)

    def predictTest(self,testSet):
        self.prediction = self.model.transform(testSet)
        # prints Accuracy
        print(bcolors.OKGREEN + "\n\nAccuracy = ", self.__getAccuracy(), '%\n\n' + bcolors.ENDC)

        # prints predicted number of faults
        self.__showNumOfPredictedFaults()

    def predict(self):
        self.prediction = self.model.transform(self.data)
        res = self.prediction.select('prediction').collect()
        if res[0][0] == 1:
            print(bcolors.ENDC,bcolors.WARNING,self.login_id, 'is may be going to have a fault today')
        else:
            print(bcolors.ENDC,bcolors.WARNING,self.login_id, 'may not have fault')

    def __showNumOfPredictedFaults(self):
        a = self.prediction.groupBy('prediction').count()
        a = a.filter(a['prediction'] == 1.0).select('count')
        a = a.withColumnRenamed('count','Predicted faults')
        a.show()
        print('\n\nActual faults =',self.__actualFault(),'\n\n')

    def __getAccuracy(self):
        evaluator = MulticlassClassificationEvaluator(labelCol=self.labelCol, predictionCol="prediction",
                                                      metricName="accuracy")
        return evaluator.evaluate(self.prediction)*100

    def __actualFault(self):
        return self.prediction.groupBy('fault').count().collect()[0][1]

    def __actualPredictedFault(self):
        return self.prediction.groupBy('prediction').count().collect()[1][1]

    def __getActiveUsers(self,session,zone):
        zones = session.read.csv('zones.csv', inferSchema=True, header=True)
        return zones.select(zone).collect()[0][0]


    def __setIndexed(self,input ,output):
        indexer = StringIndexer(inputCol=input, outputCol=output)
        self.data = indexer.fit(self.data).transform(self.data)

    def __saveModel(self,model):
        model.write().overwrite().save(self.modelFile)

    def __setFeatureCols(self,featureCols, output):
        self.inputCols = featureCols
        self.featureCol = output
        self.assembler = VectorAssembler(inputCols=featureCols, outputCol=output)
        self.data = self.assembler.transform(self.data)

    def __setLabelCol(self,label):
        self.labelCol = label


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'