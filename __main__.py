from Cep_NFP import Cep_NFP,bcolors

def main():

    # create object of Library
    cep = Cep_NFP()

    # create spark session
    # spark = cep.createSparkSession(sessionName='My_Session')

    # opens csv file ## avoid when using real time
    cep.openCSVFile(path='data.csv')


    # zone_index = 'indexed value of zone'
    # speed_index = 'indexed value of speed'
    # duration = 'difference between on that day and session start day (unit = day)'
    # downloaded = 'downloaded num of bytes during the session'
    # uploaded = 'uploaded num of bytes during the session'
    # active_in_zone = 'num of active user in the zone'
    # creates 'feature' col by Vector Assembler
    '''
    # only use when using in REAL TIME
    active_in_zone = cep.getActiveUsers(spark,'ZONE NAME')
    cep.createDataframe('zone','speed','duration(int)','downloaded(int)','uploaded(int)','active_in_zone(int)')
    
    '''

    # splits data (70% - training set, 30% - testing set)
    (trainingData, testData) = cep.getSplitedData(0.7, 0.3)

    # checks whether MODEL is trained or not
    if cep.getModel() == None :
        cep.initRandomForest()
        cep.trainModel(trainSet=trainingData) # trains the MODEL and saves to directory

    # predicts with test data set # avoid in REAL TIME
    cep.predictTest(testSet=testData)


    '''
    #only use when REAL TIME
    
    cep.predict()
    '''




if __name__ == '__main__':
    main()
