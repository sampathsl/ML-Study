rm(list=ls())

library(readxl)
############################# WITH SALARY ##############################

setwd("C:/Users/azeema/Desktop/Azeem/MSc/DS/PRoject/reaquadataset")
lap <-  read_excel("Final-A.A.A.xlsx", sheet = "Female-With_Sal6")

for(i in 1:length(lap$`Terminated (6 Months)`)){
  if(lap$`Terminated (6 Months)`[i]=="Not Terminated"){
    lap$`Terminated (6 Months)`[i]="N"
  }else if(lap$`Terminated (6 Months)`=="Terminated within 6 months"){
    lap$`Terminated (6 Months)`[i]="T"
  }
}

lap$epf=as.factor(lap$EPF_Number__t2o)
lap$EPF_Number__t2o=NULL
lap$permresid = as.factor(lap$PermanentRecidence_cat)
lap$PermanentRecidence_cat = NULL
lap$resid = as.factor(lap$Recidence)
lap$Recidence = NULL
lap$civil = as.factor(lap$CivilStatus_cat)
lap$CivilStatus_cat = NULL
lap$edu = as.factor(lap$HighestEducationQualification_cat)
lap$HighestEducationQualification_cat = NULL
lap$extracuricuar = as.factor(lap$ExtraCurricularActivities)
lap$ExtraCurricularActivities = NULL
lap$vocquali = as.factor(lap$ApparelRelatedVocationalQualification)
lap$ApparelRelatedVocationalQualification = NULL
lap$prevjob = as.factor(lap$PreviousJob)
lap$PreviouseJob = NULL
lap$expsec = as.factor(lap$ExperiencedSection_cat)
lap$ExperiencedSection_cat = NULL
lap$relaive = as.factor(lap$RelativesInApparel)
lap$RelativesInApparel = NULL
lap$spouse = as.factor(lap$SpouseOccupation_cat)
lap$SpouseOccupation_cat = NULL
lap$famop = as.factor(lap$FamilyOppinionAboutTheJob)
lap$FamilyOppinionAboutTheJob = NULL
lap$ref = as.factor(lap$Referel_cat)
lap$Referel_cat = NULL
lap$expjob = as.factor(lap$ExpectationOfDoingTheJob_cat)
lap$ExpectationOfDoingTheJob_cat = NULL
lap$transport = as.factor(lap$AvailabilityOfTransportNearTheResidence)
lap$AvailabilityOfTransportNearTheResidence = NULL
lap$reasonapp = as.factor(lap$ReasonForChooseApparel)
lap$ReasonForChooseApparel = NULL
lap$prevwork = as.factor(lap$PreviousWorkPlace)
lap$PreviousWorkPlace = NULL
lap$conrib = as.factor(lap$ContributionToTheFamilyIncome_cat)
lap$ContributionToTheFamilyIncome_cat = NULL
lap$persimp = as.factor(lap$PersonalImpression)
lap$PersonalImpression = NULL
lap$retcat = as.factor(lap$RetentionCategory)
lap$RetentionCategory= NULL
lap$seldept = as.factor(lap$SelectedDepartment)
lap$SelectedDepartment = NULL
lap$child = as.factor(lap$Children_cat)
lap$Children_cat = NULL

lap$appexp = as.factor(lap$ApparelExperience)
lap$ApparelExperience = NULL
lap$resonforleaving = as.factor(lap$ReasonForLeaving_cat)
lap$ReasonForLeaving_cat = NULL
lap$height = as.numeric(lap$Height)
lap$Height = NULL
lap$weight = as.numeric(lap$Weight)
lap$Weight = NULL
lap$grossal = as.numeric(lap$`2nd Month Gross Salary`)
lap$`2nd Month Gross Salary` = NULL
lap$basicearn = as.numeric(lap$`2nd Month Basic Pay`)
lap$`2nd Month Basic Pay` = NULL
lap$Overtime= as.numeric(lap$`2nd Month OT`)
lap$`2nd Month OT` = NULL
lap$incentive = as.numeric(lap$`2nd Month Incentive`)
lap$`2nd Month Incentive` = NULL
lap$lastsal = as.factor(lap$LastBasicSal_cat)
lap$LastBasicSal_cat = NULL
lap$medtest = as.factor(lap$MedicalTest)
lap$MedicalTest = NULL
lap$iqtest = as.numeric(lap$IQTestScore)
lap$IQTestScore = NULL
lap$expsal = as.numeric(lap$ExpectedSalary)
lap$ExpectedSalary = NULL
lap$folowincoursesext = as.factor(lap$FollowingExternalCourses)
lap$FollowingExternalCourses = NULL
lap$interviewby = as.factor(lap$InterviewedBy)
lap$InterviewedBy = NULL
lap$age = as.numeric(lap$`Age(Median)`)
lap$`Age(Median)` = NULL
lap$Terminated_Date__c = NULL
lap$Date_Joined = NULL
lap$PreviousJob=NULL
lap$Age_cat = NULL
lap$Age_Joined = NULL
lap$Date_Joined__t2o = NULL
lap$Terminated_Date__t2o = NULL
lap$`Total months worked` = NULL
lap$`Total Days` = NULL
lap$NoPay = as.numeric(lap$`2nd Month No Pay Days`)
lap$`2nd Month No Pay Days` = NULL
lap$y = as.factor(lap$`Terminated (6 Months)`)
lap$`Terminated (6 Months)` = NULL
lap$`Terminated (3 Months)`=NULL
lap$TIme_3=NULL
lap$Survival_3 = NULL

lap$Survival_6 = NULL
lap$Time_6 = NULL

lap$`3rd Month Gross Salary` = NULL
lap$DateJoined_Use6 = NULL
lap$Term6_Use = NULL
lap$Gender = NULL


names(lap)
str(lap)
lanp = na.omit(lap)
table(lanp$y)

#install.packages("randomForest")
library("randomForest")
set.seed(100)
rf=randomForest(y~.,data=lanp[,-1],importance=T)
imp=data.frame(rf$importance)
varImpPlot(rf)
#names(rf)

#Save Model Female With Salary
saveRDS(rf,"C:/Users/azeema/Desktop/Azeem/MSc/DS/PRoject/reaquadataset/RanForFemaleWithSal.rds")


rf$confusion
#tenp <- rbind(lanp[1, ] , tenp)
#tenp <- tenp[-1,]



#Testing  
tep <-  read_excel("Final-A.A.A.xlsx", sheet = "Test6-Wit Sal")

for(i in 1:length(tep$`Terminated (6 Months)`)){
  if(tep$`Terminated (6 Months)`[i]=="Not Terminated"){
    tep$`Terminated (6 Months)`[i]="N"
  }else if(tep$`Terminated (6 Months)`=="Terminated within 6 months"){
    tep$`Terminated (6 Months)`[i]="T"
  }
}

tep$epf = as.factor(tep$EPF_Number__t2o)
tep$EPF_Number__t2o=NULL
#tep$epf=NULL
tep$permresid = as.factor(tep$PermanentRecidence_cat)
tep$PermanentRecidence_cat = NULL
tep$resid = as.factor(tep$Recidence)
tep$Recidence = NULL
tep$civil = as.factor(tep$CivilStatus_cat)
tep$CivilStatus_cat = NULL
tep$edu = as.factor(tep$HighestEducationQualification_cat)
tep$HighestEducationQualification_cat = NULL
tep$extracuricuar = as.factor(tep$ExtraCurricularActivities)
tep$ExtraCurricularActivities = NULL
tep$vocquali = as.factor(tep$ApparelRelatedVocationalQualification)
tep$ApparelRelatedVocationalQualification = NULL
tep$prevjob = as.factor(tep$PreviousJob)
tep$PreviouseJob = NULL
tep$expsec = as.factor(tep$ExperiencedSection_cat)
tep$ExperiencedSection_cat = NULL
tep$relaive = as.factor(tep$RelativesInApparel)
tep$RelativesInApparel = NULL
tep$spouse = as.factor(tep$SpouseOccupation_cat)
tep$SpouseOccupation_cat = NULL
tep$famop = as.factor(tep$FamilyOppinionAboutTheJob)
tep$FamilyOppinionAboutTheJob = NULL

tep$ref = as.factor(tep$Referel_cat)
tep$Referel_cat = NULL
tep$expjob = as.factor(tep$ExpectationOfDoingTheJob_cat)
tep$ExpectationOfDoingTheJob_cat = NULL
tep$transport = as.factor(tep$AvailabilityOfTransportNearTheResidence)
tep$AvailabilityOfTransportNearTheResidence = NULL
tep$reasonapp = as.factor(tep$ReasonForChooseApparel)

tep$ReasonForChooseApparel = NULL
tep$prevwork = as.factor(tep$PreviousWorkPlace)
tep$PreviousWorkPlace = NULL
tep$conrib = as.factor(tep$ContributionToTheFamilyIncome_cat)
tep$ContributionToTheFamilyIncome_cat = NULL
tep$persimp = as.factor(tep$PersonalImpression)
tep$PersonalImpression = NULL
tep$retcat = as.factor(tep$RetentionCategory)
tep$RetentionCategory= NULL
tep$seldept = as.factor(tep$SelectedDepartment)
tep$SelectedDepartment = NULL
tep$child = as.factor(tep$Children_cat)
tep$Children_cat = NULL

tep$appexp = as.factor(tep$ApparelExperience)
tep$ApparelExperience = NULL
tep$resonforleaving = as.factor(tep$ReasonForLeaving_cat)
tep$ReasonForLeaving_cat = NULL
tep$height = as.numeric(tep$Height)
tep$Height = NULL
tep$weight = as.numeric(tep$Weight)
tep$Weight = NULL
tep$grossal = as.numeric(tep$`2nd Month Gross Salary`)
tep$`2nd Month Gross Salary` = NULL
tep$basicearn = as.numeric(tep$`2nd Month Basic Pay`)
tep$`2nd Month Basic Pay` = NULL
tep$Overtime= as.numeric(tep$`2nd Month OT`)
tep$`2nd Month OT` = NULL
tep$incentive = as.numeric(tep$`2nd Month Incentive`)
tep$`2nd Month Incentive` = NULL
tep$lastsal = as.factor(tep$LastBasicSal_cat)
tep$LastBasicSal_cat = NULL
tep$medtest = as.factor(tep$MedicalTest)
tep$MedicalTest = NULL
tep$iqtest = as.numeric(tep$IQTestScore)
tep$IQTestScore = NULL
tep$expsal = as.numeric(tep$ExpectedSalary)
tep$ExpectedSalary = NULL
tep$folowincoursesext = as.factor(tep$FollowingExternalCourses)
tep$FollowingExternalCourses = NULL
tep$interviewby = as.factor(tep$InterviewedBy)
tep$InterviewedBy = NULL
tep$age = as.numeric(tep$`Age(Median)`)
tep$`Age(Median)` = NULL
tep$Terminated_Date__c = NULL
tep$Date_Joined = NULL
tep$PreviousJob=NULL
tep$Age_cat = NULL
tep$Age_Joined = NULL
tep$Date_Joined__t2o = NULL

tep$Terminated_Date__t2o = NULL
tep$`Total months worked` = NULL
tep$`Total Days` = NULL
tep$NoPay = as.numeric(tep$`2nd Month No Pay Days`)
tep$`2nd Month No Pay Days` = NULL
tep$y = as.factor(tep$`Terminated (6 Months)`)
tep$`Terminated (6 Months)` = NULL
tep$`Terminated (3 Months)`=NULL
tep$TIme_3=NULL
tep$Survival_3 = NULL

tep$Survival_6 = NULL
tep$Time_6 = NULL

tep$`3rd Month Gross Salary` = NULL
tep$DateJoined_Use6 = NULL
tep$Term6_Use = NULL
tep$Gender = NULL

names(tep)
str(tep)
tenp = na.omit(tep)
table(tenp$y)
#names(tenp)
#names(lanp)

#View(tenp)
#ncol(lanp)
#ncol(tenp)

tenp <- rbind(lanp[1, ], tenp)
tenp <- tenp[-1,]

# predictions 
pred_forest_sal=predict(rf,newdata = tenp, type = "response")
table(pred_forest_sal,tenp$y)

rsal=data.frame(epf=factor(),predicted=factor())
p=as.data.frame(pred_forest_sal)
rsal=cbind(tenp$epf,p)
rsal=as.data.frame(rsal)


library(writexl)
write_xlsx(rsal,"C:/Users/azeema/Desktop/Azeem/MSc/DS/PRoject/reaquadataset/batch_5_3-rf-ws.xlsx")

