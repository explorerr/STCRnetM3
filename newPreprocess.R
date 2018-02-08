library(caret)
library(data.table)
library(lubridate)

#############################################
###Load the data
#############################################

dt  <- fread("M3.csv", integer64="numeric", na.strings=c("NULL","NA", "", "\\N"), header=T)
setnames(dt, c("Starting Year", "Starting Month"), c("startYear", "startMonth"))

###For those with year, just set startYear==1990 and startMonth==1
dt[startYear==0, startYear:=1990]
dt[startMonth==0, startMonth:=1]

###Transfer the data to a data point a line for ML training
newDT   <- melt(dt, id=c('Series', 'N','NF', 'Category', 'startYear', 'startMonth'))
newDT   <- newDT[!is.na(value),]
newDT   <- newDT[order(Series, variable),]

### Mark the year and month of the data
setnames(newDT, c('variable','value'), c('iMonth', 'qtty'))
newDT[, iMonth:=as.numeric(iMonth)]
newDT[, month:= iMonth %%12 ]; newDT[month==0, month:=12]
newDT[, year:=startYear+(iMonth-1)%/%12]

###Take the last 18 data points as the test dataset: mark==2
newDT[, mark:=0]
newDT[iMonth+NF-N>0, mark:=2]

########################
# For every year and the testing set, take 12 month as a period
# (1) For every test dataset, take the lag 12 month as a period, generated the lagMedian12, lagMedian6, lagMedian3, lagMedian1,
    #lagMean12, lagMean6, lagMean3
########################

##Split the time into multiple period: test, lag12, lag13-24, lag25-36...
newDT[,periodIndex:=1]
seriesList  <- unique(newDT$Series)
periodDataList  <- lapply(seriesList, function(x){
    subData     <- newDT[Series==x, ]
    ##The length of trainning data points
    trainLen    <- unique(subData$N)-unique(subData$NF)
    K   <- trainLen%/%12
    if(trainLen %%12>0){K   <- K+1}
    for(i in 1:K){
        subIndex    <- trainLen-12*i+(1:12)
        subData[iMonth %in% subIndex, periodIndex:=i+1]
    }
    subData
})
newDT2  <- rbindlist(periodDataList)

newDT2[, c("lagMedian12", "lagMean12"):=list(median(qtty),mean(qtty)), by=c("Series", "periodIndex")]
newDT2[, c("lagMedianFirst3", "lagMeanFirst3"):=list(median(head(qtty,3)), mean(head(qtty,3))), by=c("Series", "periodIndex")]
newDT2[, c("lagMedian6", "lagMean6"):=list(median(tail(qtty,6)), mean(tail(qtty,6))), by=c("Series", "periodIndex")]
newDT2[, c("lagMedian4", "lagMean4"):=list(median(tail(qtty,4)), mean(tail(qtty,4))), by=c("Series", "periodIndex")]
newDT2[, c("lagMedian3", "lagMean3"):=list(median(tail(qtty,3)), mean(tail(qtty,3))), by=c("Series", "periodIndex")]
newDT2[, lagMean2:=mean(tail(qtty,2)), by=c("Series", "periodIndex")]
newDT2[, lagMedian1:=tail(qtty,1), by=c("Series", "periodIndex")]

lagFeatures  <- c("lagMedian12", "lagMean12","lagMedianFirst3","lagMeanFirst3",
                     "lagMedian6","lagMean6","lagMedian4","lagMean4","lagMedian3","lagMean3","lagMean2","lagMedian1")
lagFeaturesData <- subset(newDT2, select=c("Series", "periodIndex",lagFeatures))
lagFeaturesData[,periodIndex:=periodIndex-1]
lagFeaturesData <- unique(lagFeaturesData, by=c("Series", "periodIndex"))

selectFeatures  <- colnames(newDT2)[!colnames(newDT2) %in% lagFeatures]
newDT2          <- subset(newDT2, select=selectFeatures)
newDT3  <- merge(newDT2, lagFeaturesData, by=c("Series", "periodIndex"), all.x=T)
newDT3  <- newDT3[!is.na(lagMedian12),]

##select the validation dataset random sample 20% of the last year's data
newDT3[, sampleIndex:=1:nrow(newDT3)]
validTotal  <- newDT3[periodIndex %in% c(2),]
validIndex  <- createDataPartition(validTotal$Series, p = .8, list = FALSE, times = 1)
validData   <- validTotal[-validIndex,] 
newDT3[sampleIndex %in% validData$sampleIndex, mark:=1]

##The ith prediction
newDT3      <- newDT3[order(Series, iMonth),]

write.table(newDT3, file='modelData.csv', sep=',', row.names=F)
