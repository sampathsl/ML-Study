rm(list=ls())

library(readxl)
setwd("C:/Users/azeema/Desktop/Azeem/MSc/DS/PRoject/reaquadataset")
lap <-  read_excel("Final-A.A.A.xlsx", sheet = "Female-With_Sal6")

for(i in 1:length(lap$`Terminated (6 Months)`)){
  if(lap$`Terminated (6 Months)`[i]=="Not Terminated"){
    lap$`Terminated (6 Months)`[i]="N"
  }else if(lap$`Terminated (6 Months)`=="Terminated within 6 months"){
    lap$`Terminated (6 Months)`[i]="T"
  }
}

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
lap$NoPay = as.numeric(lap$`2nd Month No Pay Days`)
lap$`2nd Month No Pay Days` = NULL

str(lap)
lanp = na.omit(lap)
table(lanp$y)

#1st analysis was conducted to 'lanp' which included all the variables 
set.seed(500)

#Obtaining traing and testing sets
#sam = sample(715,715*.8,replace = FALSE)
#length(sam)
#trp = lanp[sam,]
#tep = lanp[-sam,]
#table(trp$y)
#table(tep$y)

#install.packages("rpart")
#install.packages("rpart.plot")
#install.packages("rattle")

library(tree)
attach(lanp)
tree1=tree(y~.,data=lanp)
plot(tree1)
text(tree1,pretty = 0)
summary(tree1)

library("rpart")
library("rpart.plot")

tree2 <- rpart(y~., data = lanp, method = "class")

library(rattle)
library(RColorBrewer)
fancyRpartPlot(tree2,compress=T,cex=0.5,type=2)

#rel_error + xstd < xerror (Decision Criteria)
plotcp(tree2)
#The line drawn by the plotcp represents the highest cross-validated error less than 
#the minimum cross-validated error plus the standard deviation of the error at that tree.

prp(tree2, extra = 101,cex=0.7, type = 0,fallen.leaves = T, box.palette = "BuRd" ,
    branch.lty = 2)

#Tree Pruning
pruned=prune(tree2, cp = 0.011)
fancyRpartPlot(pruned,cex=0.5, uniform =T ,compress=T)
prp(pruned, extra = 101,cex=0.7, type = 0,fallen.leaves = T, box.palette = "BuRd" ,
    branch.lty = 2)
par(mfrow = c(1, 1))

#Save Model Female With Salary
saveRDS(pruned,"C:/Users/azeema/Desktop/Azeem/MSc/DS/PRoject/reaquadataset/FemaleWithSal.rds")


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
tep$NoPay = as.numeric(tep$`2nd Month No Pay Days`)
tep$`2nd Month No Pay Days` = NULL

str(tep)
tenp = na.omit(tep)
table(tenp$y)


pred2=predict(pruned,newdata = tenp,type = "class")
length(pred2)


table(pred2)
rsal=data.frame(epf=factor(),predicted=factor())
p=as.data.frame(pred2)
rsal=cbind(tenp$epf,p)
rsal=as.data.frame(rsal)


#install.packages("writexl")
library(writexl)
write_xlsx(rsal,"C:/Users/azeema/Desktop/Azeem/MSc/DS/PRoject/reaquadataset/DecisionTree_witsal.xlsx")

#tep.accuracy2=(97)/(97+46)


####################### Without Salary ##########################
lap1 <-  read_excel("Final-A.A.A.xlsx", sheet =  "Female-Without_Sal6")

for(i in 1:length(lap1$`Terminated (6 Months)`)){
  if(lap1$`Terminated (6 Months)`[i]=="Not Terminated"){
    lap1$`Terminated (6 Months)`[i]="N"
  }else if(lap1$`Terminated (6 Months)`=="Terminated within 6 months"){
    lap1$`Terminated (6 Months)`[i]="T"
  }
}

lap1$EPF_Number__t2o=NULL
lap1$permresid = as.factor(lap1$PermanentRecidence_cat)
lap1$PermanentRecidence_cat = NULL
lap1$resid = as.factor(lap1$Recidence)
lap1$Recidence = NULL
lap1$civil = as.factor(lap1$CivilStatus_cat)
lap1$CivilStatus_cat = NULL
lap1$edu = as.factor(lap1$HighestEducationQualification_cat)
lap1$HighestEducationQualification_cat = NULL
lap1$extracuricuar = as.factor(lap1$ExtraCurricularActivities)
lap1$ExtraCurricularActivities = NULL
lap1$vocquali = as.factor(lap1$ApparelRelatedVocationalQualification)
lap1$ApparelRelatedVocationalQualification = NULL
lap1$prevjob = as.factor(lap1$PreviousJob)
lap1$PreviouseJob = NULL
lap1$expsec = as.factor(lap1$ExperiencedSection_cat)
lap1$ExperiencedSection_cat = NULL
lap1$relaive = as.factor(lap1$RelativesInApparel)
lap1$RelativesInApparel = NULL
lap1$spouse = as.factor(lap1$SpouseOccupation_cat)
lap1$SpouseOccupation_cat = NULL
lap1$famop = as.factor(lap1$FamilyOppinionAboutTheJob)
lap1$FamilyOppinionAboutTheJob = NULL
lap1$ref = as.factor(lap1$Referel_cat)
lap1$Referel_cat = NULL
lap1$expjob = as.factor(lap1$ExpectationOfDoingTheJob_cat)
lap1$ExpectationOfDoingTheJob_cat = NULL
lap1$transport = as.factor(lap1$AvailabilityOfTransportNearTheResidence)
lap1$AvailabilityOfTransportNearTheResidence = NULL
lap1$reasonapp = as.factor(lap1$ReasonForChooseApparel)
lap1$ReasonForChooseApparel = NULL
lap1$prevwork = as.factor(lap1$PreviousWorkPlace)
lap1$PreviousWorkPlace = NULL
lap1$conrib = as.factor(lap1$ContributionToTheFamilyIncome_cat)
lap1$ContributionToTheFamilyIncome_cat = NULL
lap1$persimp = as.factor(lap1$PersonalImpression)
lap1$PersonalImpression = NULL
lap1$retcat = as.factor(lap1$RetentionCategory)
lap1$RetentionCategory= NULL
lap1$seldept = as.factor(lap1$SelectedDepartment)
lap1$SelectedDepartment = NULL
lap1$child = as.factor(lap1$Children_cat)
lap1$Children_cat = NULL

lap1$appexp = as.factor(lap1$ApparelExperience)
lap1$ApparelExperience = NULL
lap1$resonforleaving = as.factor(lap1$ReasonForLeaving_cat)
lap1$ReasonForLeaving_cat = NULL
lap1$height = as.numeric(lap1$Height)
lap1$Height = NULL
lap1$weight = as.numeric(lap1$Weight)
lap1$Weight = NULL
lap1$lastsal = as.factor(lap1$LastBasicSal_cat)
lap1$LastBasicSal_cat = NULL
lap1$medtest = as.factor(lap1$MedicalTest)
lap1$MedicalTest = NULL
lap1$iqtest = as.numeric(lap1$IQTestScore)
lap1$IQTestScore = NULL
lap1$expsal = as.numeric(lap1$ExpectedSalary)
lap1$ExpectedSalary = NULL
lap1$folowincoursesext = as.factor(lap1$FollowingExternalCourses)
lap1$FollowingExternalCourses = NULL
lap1$interviewby = as.factor(lap1$InterviewedBy)
lap1$InterviewedBy = NULL
lap1$age = as.numeric(lap1$`Age(Median)`)
lap1$`Age(Median)` = NULL
lap1$Terminated_Date__c = NULL
lap1$Date_Joined = NULL
lap1$PreviousJob=NULL
lap1$Age_cat = NULL
lap1$Age_Joined = NULL
lap1$Date_Joined__t2o = NULL
lap1$Terminated_Date__t2o = NULL
lap1$`Total months worked` = NULL
lap1$`Total Days` = NULL

lap1$y = as.factor(lap1$`Terminated (6 Months)`)
lap1$`Terminated (6 Months)` = NULL
lap1$`Terminated (3 Months)`=NULL
lap1$TIme_3=NULL
lap1$Survival_3 = NULL

lap1$Survival_6 = NULL
lap1$Time_6 = NULL

lap1$`3rd Month Gross Salary` = NULL
lap1$DateJoined_Use6 = NULL
lap1$Term6_Use = NULL
lap1$Gender = NULL

str(lap1)
lanp1 = na.omit(lap1)
table(lanp1$y)

#2nd analysis was conducted to 'lanp1' excluding salary variables 
set.seed(500)

#Obtaining traing and testing sets
#sam = sample(715,715*.8,replace = FALSE)
#length(sam)
#trp1 = lanp1[sam,]
#tep1 = lanp1[-sam,]
#table(trp1$y)
#table(tep1$y)

library(tree)
tree3=tree(y~.,data=lanp1)
plot(tree3)
text(tree3,pretty = 0)
summary(tree3)

#pred3=predict(tree3,newdata = tep1,type="class")
#length(pred3)
#table(pred3,tep1$y)
#table(tep1$y)

library("rpart")
library("rpart.plot")

tree4 <- rpart(y~., data = lanp1, method = "class")

library(rattle)
library(RColorBrewer)

fancyRpartPlot(tree4,compress=T,cex=0.5)

#rel_error + xstd < xerror (Decision Criteria)
plotcp(tree4)
#The line drawn by the plotcp represents the highest cross-validated error less than 
#the minimum cross-validated error plus the standard deviation of the error at that tree.
plot(tree4, uniform=TRUE, main="Classification Tree for Linea Aqua ")
text(tree4, pretty =  0)
#summary(treela)

#Tree Pruning
pruned1=prune(tree4, cp = 0.015)
fancyRpartPlot(pruned1,cex=0.5, uniform =T ,compress=T)
prp(pruned1, extra = 101,cex=0.7, type = 0,fallen.leaves = T, box.palette = "BuRd" ,
    branch.lty = 2)
par(mfrow = c(1, 1))

#Save Model Female Without Salary
saveRDS(pruned1,"C:/Users/azeema/Desktop/Azeem/MSc/DS/PRoject/reaquadataset/FemaleWithoutSal.rds")



tep1 <-  read_excel("Final-A.A.A.xlsx", sheet =  "Test6-Without Sal")

for(i in 1:length(tep1$`Terminated (6 Months)`)){
  if(tep1$`Terminated (6 Months)`[i]=="Not Terminated"){
    tep1$`Terminated (6 Months)`[i]="N"
  }else if(tep1$`Terminated (6 Months)`=="Terminated within 6 months"){
    tep1$`Terminated (6 Months)`[i]="T"
  }
}

tep1$epf = tep1$EPF_Number__t2o
tep1$EPF_Number__t2o=NULL
tep1$permresid = as.factor(tep1$PermanentRecidence_cat)
tep1$PermanentRecidence_cat = NULL
tep1$resid = as.factor(tep1$Recidence)
tep1$Recidence = NULL
tep1$civil = as.factor(tep1$CivilStatus_cat)
tep1$CivilStatus_cat = NULL
tep1$edu = as.factor(tep1$HighestEducationQualification_cat)
tep1$HighestEducationQualification_cat = NULL
tep1$extracuricuar = as.factor(tep1$ExtraCurricularActivities)
tep1$ExtraCurricularActivities = NULL
tep1$vocquali = as.factor(tep1$ApparelRelatedVocationalQualification)
tep1$ApparelRelatedVocationalQualification = NULL
tep1$prevjob = as.factor(tep1$PreviousJob)
tep1$PreviouseJob = NULL
tep1$expsec = as.factor(tep1$ExperiencedSection_cat)
tep1$ExperiencedSection_cat = NULL
tep1$relaive = as.factor(tep1$RelativesInApparel)
tep1$RelativesInApparel = NULL
tep1$spouse = as.factor(tep1$SpouseOccupation_cat)
tep1$SpouseOccupation_cat = NULL
tep1$famop = as.factor(tep1$FamilyOppinionAboutTheJob)
tep1$FamilyOppinionAboutTheJob = NULL
tep1$ref = as.factor(tep1$Referel_cat)
tep1$Referel_cat = NULL
tep1$expjob = as.factor(tep1$ExpectationOfDoingTheJob_cat)
tep1$ExpectationOfDoingTheJob_cat = NULL
tep1$transport = as.factor(tep1$AvailabilityOfTransportNearTheResidence)
tep1$AvailabilityOfTransportNearTheResidence = NULL
tep1$reasonapp = as.factor(tep1$ReasonForChooseApparel)
tep1$ReasonForChooseApparel = NULL
tep1$prevwork = as.factor(tep1$PreviousWorkPlace)
tep1$PreviousWorkPlace = NULL
tep1$conrib = as.factor(tep1$ContributionToTheFamilyIncome_cat)
tep1$ContributionToTheFamilyIncome_cat = NULL
tep1$persimp = as.factor(tep1$PersonalImpression)
tep1$PersonalImpression = NULL
tep1$retcat = as.factor(tep1$RetentionCategory)
tep1$RetentionCategory= NULL
tep1$seldept = as.factor(tep1$SelectedDepartment)
tep1$SelectedDepartment = NULL
tep1$child = as.factor(tep1$Children_cat)
tep1$Children_cat = NULL

tep1$appexp = as.factor(tep1$ApparelExperience)
tep1$ApparelExperience = NULL
tep1$resonforleaving = as.factor(tep1$ReasonForLeaving_cat)
tep1$ReasonForLeaving_cat = NULL
tep1$height = as.numeric(tep1$Height)
tep1$Height = NULL
tep1$weight = as.numeric(tep1$Weight)
tep1$Weight = NULL
tep1$lastsal = as.factor(tep1$LastBasicSal_cat)
tep1$LastBasicSal_cat = NULL
tep1$medtest = as.factor(tep1$MedicalTest)
tep1$MedicalTest = NULL
tep1$iqtest = as.numeric(tep1$IQTestScore)
tep1$IQTestScore = NULL
tep1$expsal = as.numeric(tep1$ExpectedSalary)
tep1$ExpectedSalary = NULL
tep1$folowincoursesext = as.factor(tep1$FollowingExternalCourses)
tep1$FollowingExternalCourses = NULL
tep1$interviewby = as.factor(tep1$InterviewedBy)
tep1$InterviewedBy = NULL
tep1$age = as.numeric(tep1$`Age(Median)`)
tep1$`Age(Median)` = NULL
tep1$Terminated_Date__c = NULL
tep1$Date_Joined = NULL
tep1$PreviousJob=NULL
tep1$Age_cat = NULL
tep1$Age_Joined = NULL
tep1$Date_Joined__t2o = NULL
tep1$Terminated_Date__t2o = NULL
tep1$`Total months worked` = NULL
tep1$`Total Days` = NULL

tep1$y = as.factor(tep1$`Terminated (6 Months)`)
tep1$`Terminated (6 Months)` = NULL
tep1$`Terminated (3 Months)`=NULL
tep1$TIme_3=NULL
tep1$Survival_3 = NULL

tep1$Survival_6 = NULL
tep1$Time_6 = NULL

tep1$`3rd Month Gross Salary` = NULL
tep1$DateJoined_Use6 = NULL
tep1$Term6_Use = NULL
tep1$Gender = NULL

str(tep1)
tenp1 = na.omit(tep1)
table(tenp1$y)


pred4=predict(pruned1,newdata = tenp1,type = "class")

#Removing level "TMPacking" from seldept which is new level
which(tenp1$seldept=="TMPacking")
tenpn=tenp1[-131,]

pred4=predict(pruned1,newdata = tenpn,type = "class")
length(pred4)

table(pred4)
rsal1=data.frame(epf=factor(),predicted=factor())
p1=as.data.frame(pred4)
rsal1=cbind(tenpn$epf,p1)
rsal1=as.data.frame(rsal1)

library(writexl)
write_xlsx(rsal1,"C:/Users/azeema/Desktop/Azeem/MSc/DS/PRoject/reaquadataset/DecisionTree_witoutsal.xlsx")

#table(pred4,tep1$y)
#table(tep1$y)
#tep.accuracy4=(123+84)/(123+84+139+42)




##############To be Done ##########################################################
################ Random Forest ####################################################
#install.packages("randomForest")
library(randomForest)
set.seed(100)
rf=randomForest(y~.,data=trp,importance=T)
imp=data.frame(rf$importance)
varImpPlot(rf)
#names(rf)
rf$confusion
pred_forest_sal=predict(rf,newdata = tep,type="response")
table(pred_forest_sal,tep$y)
accu_sal=(14+92)/(14+92+37)
accu_sal
#Recall
(14)/(24)
#Precision
(14)/(14+27)


rf_withoutsal=randomForest(y~.,data=trp1)
varImpPlot(rf_withoutsal)
pred_forest=predict(rf_withoutsal,newdata = tep1,type="response")
table(pred_forest,tep1$y)
accu_sal=(188)/(188+188+171+29)
accu_sal
#Recall
91/(91+29)
#Precision
91/(91+171)


############################ Ada Boosting ########################################
#install.packages("ada")
library(ada)
#trp$resid=NULL
x_sal=model.matrix(trp$y~.,trp)[,-1]
y_sal=trp$y

#tep$resid=NULL
x_sal_test=model.matrix(tep$y~.,tep)[,-1]
y_sal_test=tep$y
new=as.data.frame(x_sal_test)
new$y=y_sal_test

set.seed(1)
ada_boost=ada(x_sal,y_sal,x_sal_test,y_sal_test,nu=0.001,iter = 100)
plot(ada_boost)
summary(ada_boost)
names(ada_boost)
ada_boost$confusion
pred_ada=predict(ada_boost,newdata = new,type="vector")
varplot(ada_boost,plot.it = T,type="scores")
table(pred_ada,tep$y)
#Accuracy
(13+86)/(13+86+16+28)
#Recall
13/(13+16)
#Precision
13/(28+13)

##Without Salary
x_withoutsal=model.matrix(lanp1$y~.,lanp1)[,-1]
y_withoutsal=lanp1$y

#tep$resid=NULL
x_test=model.matrix(tenp1$y~.,tenp1)[,-1]
y_test=tenp1$y
new=as.data.frame(x_test)
#new$y=y_sal_test

set.seed(1)
ada_boost2=ada(x_withoutsal,y_withoutsal,x_test,y_test,nu=0.001,iter = 100)
plot(ada_boost2)
summary(ada_boost2)
#names(ada_boost)
#ada_boost$confusion
pred_ada2=predict(ada_boost2,newdata = new,type="vector")
varplot(ada_boost2,plot.it = T,type="scores")
table(pred_ada2,tenp1$y)
#Accuracy
(91+93)/(91+93+171+33)
#Recall
91/(91+33)
#Precision
91/(91+171)

