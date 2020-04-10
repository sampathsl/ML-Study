library(readxl)
gg <- read_excel("C:/Users/azeema/Desktop/Azeem/MSc/DS/PRoject/reaquadataset/Final-A.A.A.xlsx", 
                 sheet = "Female-With_Sal6")
View(gg)
colnames(gg)
g<-gg

colnames(g)

sum(is.na(g))
g$Gender=NULL
g$EPF_Number__t2o=NULL
g$permresid=as.factor(g$PermanentRecidence_cat)
g$PermanentRecidence_cat=NULL
g$resi=as.factor(g$Recidence)
g$Recidence=NULL
g$civil=as.factor(g$CivilStatus_cat)
g$CivilStatus_cat=NULL
g$edu=as.factor(g$HighestEducationQualification_cat)
g$HighestEducationQualification_cat=NULL
g$inter=as.factor(g$InterviewedBy)
g$InterviewedBy=NULL
g$extra=as.factor(g$ExtraCurricularActivities)
g$ExtraCurricularActivities=NULL
g$appvoc=as.factor(g$ApparelRelatedVocationalQualification)
g$ApparelRelatedVocationalQualification=NULL
g$prevjob=as.factor(g$PreviousJob)
g$PreviousJob=NULL
g$expsec=as.factor(g$ExperiencedSection_cat)
g$ExperiencedSection_cat=NULL
g$relative=as.factor(g$RelativesInApparel)
g$RelativesInApparel=NULL
g$spouse=as.factor(g$SpouseOccupation_cat)
g$SpouseOccupation_cat=NULL
g$famop=as.factor(g$FamilyOppinionAboutTheJob)
g$FamilyOppinionAboutTheJob=NULL
g$ref=as.factor(g$Referel_cat)
g$Referel_cat=NULL
g$exp=as.factor(g$ExpectationOfDoingTheJob_cat)
g$ExpectationOfDoingTheJob_cat=NULL
g$transport=as.factor(g$AvailabilityOfTransportNearTheResidence)
g$AvailabilityOfTransportNearTheResidence=NULL
g$reasonapp=as.factor(g$ReasonForChooseApparel)
g$ReasonForChooseApparel=NULL
g$prevworkplace=as.factor(g$PreviousWorkPlace)
g$PreviousWorkPlace=NULL
g$perImp=as.factor(g$PersonalImpression)
g$PersonalImpression=NULL
g$contrib=as.factor(g$ContributionToTheFamilyIncome_cat)
g$ContributionToTheFamilyIncome_cat=NULL
g$child=as.factor(g$Children_cat)
g$Children_cat=NULL
g$appexp=as.factor(g$ApparelExperience)
g$ApparelExperience=NULL
g$resonleave=as.factor(g$ReasonForLeaving_cat)
g$ReasonForLeaving_cat=NULL
g$height=as.numeric(g$Height)
g$Height=NULL
g$weight=as.numeric(g$Weight)
g$Weight=NULL
g$grossal=as.numeric(g$`2nd Month Gross Salary`)
g$`2nd Month Gross Salary`=NULL
g$basicsal=as.numeric(g$`2nd Month Basic Pay`)
g$`2nd Month Basic Pay`=NULL
g$ot=as.numeric(g$`2nd Month OT`)
g$`2nd Month OT`=NULL
g$nopay=as.factor(g$`2nd Month No Pay Days`)
g$`2nd Month No Pay Days`=NULL
g$incent=as.numeric(g$`2nd Month Incentive`)
g$`2nd Month Incentive`=NULL
g$med=as.factor(g$MedicalTest)
g$MedicalTest=NULL
g$iq=as.factor(g$IQTestScore)
g$IQTestScore=NULL
g$expsal=as.factor(g$ExpectedSalary)
g$ExpectedSalary=NULL
g$extcourse=as.factor(g$FollowingExternalCourses)
g$FollowingExternalCourses=NULL
g$lastsal=as.factor(g$LastBasicSal_cat)
g$LastBasicSal_cat=NULL
g$Age_cat=NULL
g$age=as.factor(g$`Age(Median)`)
g$`Age(Median)`=NULL
g$Age_Joined=NULL
g$y=as.factor(g$`Terminated (6 Months)`)
g$`Terminated (6 Months)`=NULL
g$Time_6=NULL
g$Survival_6=NULL
g$`Total months worked`=NULL
g$`Total Days`=NULL
g$DateJoined_Use6=NULL
g$Term6_Use=NULL
g$`3rd Month Gross Salary`=NULL
g$retcat=as.factor(g$RetentionCategory)
g$RetentionCategory=NULL
g$seldept=as.factor(g$SelectedDepartment)
g$SelectedDepartment=NULL


View(g)
colnames(g)

sum(is.na(g))
d <- na.omit(g)
sum(is.na(d))
summary(d)
#term = 240
#notterm=475

#split data
set.seed(2)
splitd<-sample(1:2, size=nrow(d), prob = c(0.8,0.2), replace = TRUE)
traind<-d[splitd==1,];
testd<-d[splitd==2,];

View(traind)
sum(is.na(traind))
summary(traind)
nlevels(traind$med)

##############  WITH SALARY ####################
library(phonTools)

#Null model
glmNN=glm(y~1,binomial,data=traind)
summary(glmNN)
ncol(traind)

#Models with one variable
names(traind)[1:38]
glm1vv = zeros(38,2)
colnames(glm1vv) = c("Name", "BIC")

for(i in 1:38){
  x = colnames(traind)[i]
  y1= as.formula(paste("y~", x))
  glmM=glm(y1,binomial,data=traind)
  glm1vv[i,1] = x
  glm1vv[i,2] = BIC(glmM)
}

glm1vv[which.min(glm1vv[,2]),]
glm11=glm(y~grossal,binomial,data=traind)
summary(glm11)

dev11 = 2*(logLik(glm11) - logLik(glmNN))
pchisq(dev11, df=1, lower.tail=FALSE)

# MOdel with 2 variables
names(traind)[1:38]
glm2vv = zeros(38,2)
colnames(glm2vv) = c("Name", " BIC")

for(i in 1:38){
  if(i == 25){
    next
  }
  x = colnames(traind)[i]
  y1= as.formula(paste("y~grossal+", x))
  glmM=glm(y1,binomial,data=traind)
  glm2vv[i,1] = x
  glm2vv[i,2] = BIC(glmM)
}
glm2vv = glm2vv[-25,]

glm2vv[which.min(glm2vv[,2]),]
glm22=glm(y~grossal+ot,binomial,data=traind)
summary(glm22)

dev22 = 2*(logLik(glm22) - logLik(glm11))
pchisq(dev11, df=1, lower.tail=FALSE)

# model with 3 variables
names(traind)[1:38]
glm3vv = zeros(38,2)
colnames(glm3vv) = c("Name", " BIC")

for(i in 1:38){
  if(i == 25){
    next
  }
  if(i == 27){
    next
  }
  x = colnames(traind)[i]
  y1= as.formula(paste("y~grossal+ot+", x))
  glmM=glm(y1,binomial,data=traind)
  glm3vv[i,1] = x
  glm3vv[i,2] = BIC(glmM)
}
glm3vv = glm3vv[-c(25,27,36),]

glm3vv[which.min(glm3vv[,2]),]
glm33=glm(y~grossal+ot+basicsal,binomial,data=traind)
summary(glm33)

dev33 = 2*(logLik(glm33) - logLik(glm22))
pchisq(dev33, df=1, lower.tail=FALSE)


# model with 4 variables
names(traind)[1:38]
glm4vv = zeros(38,2)
colnames(glm4vv) = c("Name"," BIC")

for(i in 1:38){
  if(i == 25){
    next
  }
  if(i == 27){
    next
  }
  if(i == 26){
    next
  }  
  x = colnames(traind)[i]
  y1= as.formula(paste("y~grossal+ot+basicsal+", x))
  glmM=glm(y1,binomial,data=traind)
  glm4vv[i,1] = x
  glm4vv[i,2] = BIC(glmM)
}
glm4vv = glm4vv[-c(25,27,26),]

glm4vv[which.min(glm4vv[,2]),]
glm44=glm(y~grossal+ot+basicsal+famop,binomial,data=traind)
summary(glm44)

dev44 = 2*(logLik(glm44) - logLik(glm33))
pchisq(dev44, df=1, lower.tail=FALSE)

# model with 5 variables
names(traind)[1:38]
glm5vv = zeros(38,2)
colnames(glm5vv) = c("Name", " BIC")

for(i in 1:38){
  if(i == 25){
    next
  }
  if(i == 27){
    next
  }
  if(i == 26){
    next
  }
  if(i == 12){
    next
  }
  x = colnames(traind)[i]
  y1= as.formula(paste("y~grossal+ot+basicsal+famop+", x))
  glmM=glm(y1,binomial,data=traind)
  glm5vv[i,1] = x
  glm5vv[i,2] = BIC(glmM)
}
glm5vv = glm5vv[-c(25,27,26,12,36),]

glm5vv[which.min(glm5vv[,2]),]
glm55=glm(y~grossal+ot+basicsal+famop+extra,binomial,data=traind)
summary(glm55)

dev55 = 2*(logLik(glm55) - logLik(glm44))
pchisq(dev55, df=1, lower.tail=FALSE)

# model with 6 variables
names(traind)[1:38]
glm6vv = zeros(38,2)
colnames(glm6vv) = c("Name", " BIC")

for(i in 1:38){
  if(i == 25){
    next
  }
  if(i == 27){
    next
  }
  if(i == 26){
    next
  }
  if(i == 12){
    next
  }
  if(i == 6){
    next
  }
  x = colnames(traind)[i]
  y1= as.formula(paste("y~grossal+ot+basicsal+famop+extra+", x))
  glmM=glm(y1,binomial,data=traind)
  glm6vv[i,1] = x
  glm6vv[i,2] = BIC(glmM)
}
glm6vv = glm6vv[-c(25,27,26,12,6,36),]

glm6vv[which.min(glm6vv[,2]),]
glm66=glm(y~grossal+ot+basicsal+famop+extra+exp,binomial,data=traind)
summary(glm66)

dev66 = 2*(logLik(glm66) - logLik(glm55))
pchisq(dev66, df=1, lower.tail=FALSE)

# model with 7 variables
names(traind)[1:38]
glm7vv = zeros(38,2)
colnames(glm7vv) = c("Name", " BIC")

for(i in 1:38){
  if(i == 25){
    next
  }
  if(i == 27){
    next
  }
  if(i == 26){
    next
  }
  if(i == 12){
    next
  }
  if(i == 6){
    next
  }
  if(i == 14){
    next
  }
  x = colnames(traind)[i]
  y1= as.formula(paste("y~grossal+ot+basicsal+famop+extra+exp+", x))
  glmM=glm(y1,binomial,data=traind)
  glm7vv[i,1] = x
  glm7vv[i,2] = BIC(glmM)
}
glm7vv = glm7vv[-c(25,26,27,12,6,14,36),]

glm7vv[which.min(glm7vv[,2]),]
glm77=glm(y~grossal+ot+basicsal+famop+extra+exp+permresid,binomial,data=traind)
summary(glm77)

dev77 = 2*(logLik(glm77) - logLik(glm66))
pchisq(dev77, df=1, lower.tail=FALSE)

# model with 8 variables
names(traind)[1:38]
glm8vv = zeros(38,2)
colnames(glm8vv) = c("Name", " BIC")

for(i in 1:38){
  if(i == 25){
    next
  }
  if(i == 27){
    next
  }
  if(i == 26){
    next
  }
  if(i == 12){
    next
  }
  if(i == 6){
    next
  }
  if(i == 14){
    next
  }
  if(i == 1){
    next
  }
  x = colnames(traind)[i]
  y1= as.formula(paste("y~grossal+ot+basicsal+famop+extra+exp+permresid+", x))
  glmM=glm(y1,binomial,data=traind)
  glm8vv[i,1] = x
  glm8vv[i,2] = BIC(glmM)
}
glm8vv = glm8vv[-c(25,26,27,12,6,14,1,36),]

glm8vv[which.min(glm8vv[,2]),]
glm88=glm(y~grossal+ot+basicsal+famop+extra+exp+permresid+appexp,binomial,data=traind)
summary(glm88)

dev88 = 2*(logLik(glm88) - logLik(glm77))
pchisq(dev88, df=1, lower.tail=FALSE)

# model with 9 variables
names(traind)[1:38]
glm9vv = zeros(38,2)
colnames(glm9vv) = c("Name", " BIC")

for(i in 1:38){
  if(i == 25){
    next
  }
  if(i == 27){
    next
  }
  if(i == 26){
    next
  }
  if(i == 12){
    next
  }
  if(i == 6){
    next
  }
  if(i == 14){
    next
  }
  if(i == 1){
    next
  }
  if(i == 21){
    next
  }
  x = colnames(traind)[i]
  y1= as.formula(paste("y~grossal+ot+basicsal+famop+extra+exp+permresid+appexp+", x))
  glmM=glm(y1,binomial,data=traind)
  glm9vv[i,1] = x
  glm9vv[i,2] = BIC(glmM)
}
glm9vv = glm9vv[-c(25,26,27,12,6,14,1,36,21),]

glm9vv[which.min(glm9vv[,2]),]
glm99=glm(y~grossal+ot+basicsal+famop+extra+exp+permresid+appexp+med,binomial,data=traind)
summary(glm99)

dev99 = 2*(logLik(glm99) - logLik(glm88))
pchisq(dev99, df=1, lower.tail=FALSE)

# model with 10 variables
names(traind)[1:38]
glm10vv = zeros(38,2)
colnames(glm10vv) = c("Name", " BIC")

for(i in 1:38){
  if(i == 25){
    next
  }
  if(i == 27){
    next
  }
  if(i == 26){
    next
  }
  if(i == 12){
    next
  }
  if(i == 6){
    next
  }
  if(i == 14){
    next
  }
  if(i == 1){
    next
  }
  if(i == 21){
    next
  }
  if(i == 30){
    next
  }
  x = colnames(traind)[i]
  y1= as.formula(paste("y~grossal+ot+basicsal+famop+extra+exp+permresid+appexp+med+", x))
  glmM=glm(y1,binomial,data=traind)
  glm10vv[i,1] = x
  glm10vv[i,2] = BIC(glmM)
}
glm10vv = glm10vv[-c(25,26,27,12,6,14,1,36,21,30),]

glm10vv[which.min(glm10vv[,2]),]
glm1010=glm(y~grossal+ot+basicsal+famop+extra+exp+permresid+appexp+med+weight,binomial,data=traind)
summary(glm1010)

dev1010 = 2*(logLik(glm1010) - logLik(glm99))
pchisq(dev1010, df=1, lower.tail=FALSE)

# model with 11 variables
names(traind)[1:38]
glm11vv = zeros(38,2)
colnames(glm11vv) = c("Name", " BIC")

for(i in 1:38){
  if(i == 25){
    next
  }
  if(i == 27){
    next
  }
  if(i == 26){
    next
  }
  if(i == 12){
    next
  }
  if(i == 6){
    next
  }
  if(i == 14){
    next
  }
  if(i == 1){
    next
  }
  if(i == 21){
    next
  }
  if(i == 30){
    next
  }
  if(i == 24){
    next
  }
  x = colnames(traind)[i]
  y1= as.formula(paste("y~grossal+ot+basicsal+famop+extra+exp+permresid+appexp+med+weight+", x))
  glmM=glm(y1,binomial,data=traind)
  glm11vv[i,1] = x
  glm11vv[i,2] = BIC(glmM)
}
glm11vv = glm11vv[-c(25,26,27,12,6,14,1,36,21,30,24),]

glm11vv[which.min(glm11vv[,2]),]
glm1111=glm(y~grossal+ot+basicsal+famop+extra+exp+permresid+appexp+med+weight+relative,binomial,data=traind)
summary(glm1111)

dev1111 = 2*(logLik(glm1111) - logLik(glm1010))
pchisq(dev1111, df=1, lower.tail=FALSE)




##Final moddel - with salary
withSalary = glm(y~grossal+ot+basicsal+famop+extra+exp+permresid+appexp+med+weight,binomial,data=traind)
summary(withSalary)

#Save Model Female With Salary
saveRDS(withSalary,"C:/Users/azeema/Desktop/Azeem/MSc/DS/PRoject/reaquadataset/FemaleWithSal.rds")


## evaluate model
predictWS<-predict(withSalary, newdata=testd, type="response")
accuracyWS<-table(predictWS>.5,testd$y)
addmargins(accuracyWS)

## Accuracy
sum(diag(accuracyWS))/sum(accuracyWS)


################## WITHOUT SALARY ######################

gg1 <- read_excel("C:/Users/azeema/Desktop/Azeem/MSc/DS/PRoject/reaquadataset/Final-A.A.A.xlsx", 
                  sheet = "Female-Without_Sal6")
colnames(gg1)

gg1$Gender=NULL
gg1$EPF_Number__t2o=NULL
gg1$permresid=as.factor(gg1$PermanentRecidence_cat)
gg1$PermanentRecidence_cat=NULL
gg1$resi=as.factor(gg1$Recidence)
gg1$Recidence=NULL
gg1$civil=as.factor(gg1$CivilStatus_cat)
gg1$CivilStatus_cat=NULL
gg1$edu=as.factor(gg1$HighestEducationQualification_cat)
gg1$HighestEducationQualification_cat=NULL
gg1$inter=as.factor(gg1$InterviewedBy)
gg1$InterviewedBy=NULL
gg1$extra=as.factor(gg1$ExtraCurricularActivities)
gg1$ExtraCurricularActivities=NULL
gg1$appvoc=as.factor(gg1$ApparelRelatedVocationalQualification)
gg1$ApparelRelatedVocationalQualification=NULL
gg1$prevjob=as.factor(gg1$PreviousJob)
gg1$PreviousJob=NULL
gg1$expsec=as.factor(gg1$ExperiencedSection_cat)
gg1$ExperiencedSection_cat=NULL
gg1$relative=as.factor(gg1$RelativesInApparel)
gg1$RelativesInApparel=NULL
gg1$spouse=as.factor(gg1$SpouseOccupation_cat)
gg1$SpouseOccupation_cat=NULL
gg1$famop=as.factor(gg1$FamilyOppinionAboutTheJob)
gg1$FamilyOppinionAboutTheJob=NULL
gg1$ref=as.factor(gg1$Referel_cat)
gg1$Referel_cat=NULL
gg1$exp=as.factor(gg1$ExpectationOfDoingTheJob_cat)
gg1$ExpectationOfDoingTheJob_cat=NULL
gg1$transport=as.factor(gg1$AvailabilityOfTransportNearTheResidence)
gg1$AvailabilityOfTransportNearTheResidence=NULL
gg1$retcat=as.factor(gg1$RetentionCategory)
gg1$RetentionCategory=NULL
gg1$seldept=as.factor(gg1$SelectedDepartment)
gg1$SelectedDepartment=NULL
gg1$reasonapp=as.factor(gg1$ReasonForChooseApparel)
gg1$ReasonForChooseApparel=NULL
gg1$prevworkplace=as.factor(gg1$PreviousWorkPlace)
gg1$PreviousWorkPlace=NULL
gg1$perImp=as.factor(gg1$PersonalImpression)
gg1$PersonalImpression=NULL
gg1$contrib=as.factor(gg1$ContributionToTheFamilyIncome_cat)
gg1$ContributionToTheFamilyIncome_cat=NULL
gg1$child=as.factor(gg1$Children_cat)
gg1$Children_cat=NULL
gg1$appexp=as.factor(gg1$ApparelExperience)
gg1$ApparelExperience=NULL
gg1$resonleave=as.factor(gg1$ReasonForLeaving_cat)
gg1$ReasonForLeaving_cat=NULL
gg1$height=as.numeric(gg1$Height)
gg1$Height=NULL
gg1$weight=as.numeric(gg1$Weight)
gg1$Weight=NULL
gg1$med=as.factor(gg1$MedicalTest)
gg1$MedicalTest=NULL
gg1$iq=as.factor(gg1$IQTestScore)
gg1$IQTestScore=NULL
gg1$expsal=as.factor(gg1$ExpectedSalary)
gg1$ExpectedSalary=NULL
gg1$extcourse=as.factor(gg1$FollowingExternalCourses)
gg1$FollowingExternalCourses=NULL
gg1$lastsal=as.factor(gg1$LastBasicSal_cat)
gg1$LastBasicSal_cat=NULL
gg1$Age_cat=NULL
gg1$age=as.factor(gg1$`Age(Median)`)
gg1$`Age(Median)`=NULL
gg1$Age_Joined=NULL
gg1$y=as.factor(gg1$`Terminated (6 Months)`)
gg1$`Terminated (6 Months)`=NULL
gg1$Time_6=NULL
gg1$Survival_6=NULL
gg1$`Total months worked`=NULL
gg1$`Total Days`=NULL
gg1$DateJoined_Use6=NULL
gg1$Term6_Use=NULL

sum(is.na(gg1))
summary(gg1)
nrow(gg1)
d1 <- na.omit(gg1)
sum(is.na(d1))
nrow(d1)
summary(d1)


#split data
set.seed(2)
splitd1<-sample(1:2, size=nrow(d1), prob = c(0.8,0.2), replace = TRUE)
traind1<-d1[splitd1==1,];
testd1<-d1[splitd1==2,];
ncol(d1)
#Null model
glmN=glm(y~1,binomial,data=traind1)
summary(glmN)

#model with 1 variable
names(traind1)[1:33]
glm1v = zeros(33,2)
colnames(glm1v) = c("Name", "BIC")

for(i in 1:33){
  x = colnames(traind1)[i]
  y1= as.formula(paste("y~", x))
  glmM=glm(y1,binomial,data=traind1)
  glm1v[i,1] = x
  glm1v[i,2] = BIC(glmM)
}

glm1v[which.min(glm1v[,2]),]
glm1=glm(y~prevworkplace,binomial,data=traind1)
summary(glm1)

dev1 = 2*(logLik(glm1) - logLik(glmN))
pchisq(dev1, df=1, lower.tail=FALSE)

#model with 2 variable
names(traind1)[1:33]
glm2v = zeros(33,2)
colnames(glm2v) = c("Name", "BIC")

for(i in 1:33){
  if(i == 19){
    next
  }
  x = colnames(traind1)[i]
  y1= as.formula(paste("y~prevworkplace+", x))
  glmM=glm(y1,binomial,data=traind1)
  glm2v[i,1] = x
  glm2v[i,2] = BIC(glmM)
}

glm2v = glm2v[-c(19,33),]
glm2v[which.min(glm2v[,2]),]
glm2=glm(y~prevworkplace+weight,binomial,data=traind1)
summary(glm2)

dev2 = 2*(logLik(glm2) - logLik(glm1))
pchisq(dev2, df=1, lower.tail=FALSE)

#model with 3 variables
names(traind1)[1:33]
glm3v = zeros(33,2)
colnames(glm3v) = c("Name", "BIC")

for(i in 1:33){
  if(i == 19){
    next
  }
  if(i == 26){
    next
  }
  x = colnames(traind1)[i]
  y1= as.formula(paste("y~prevworkplace+weight+", x))
  glmM=glm(y1,binomial,data=traind1)
  glm3v[i,1] = x
  glm3v[i,2] = BIC(glmM)
}

glm3v = glm3v[-c(19,26,33),]
glm3v[which.min(glm3v[,2]),]
glm3=glm(y~prevworkplace+weight+height,binomial,data=traind1)
summary(glm3)

dev3 = 2*(logLik(glm3) - logLik(glm2))
pchisq(dev3, df=1, lower.tail=FALSE)

#model with 4 variables
names(traind1)[1:33]
glm4v = zeros(33,2)
colnames(glm4v) = c("Name", "BIC")

for(i in 1:33){
  if(i == 19){
    next
  }
  if(i == 26){
    next
  }
  if(i == 25){
    next
  }
  x = colnames(traind1)[i]
  y1= as.formula(paste("y~prevworkplace+weight+height+", x))
  glmM=glm(y1,binomial,data=traind1)
  glm4v[i,1] = x
  glm4v[i,2] = BIC(glmM)
}

glm4v = glm4v[-c(19,26,25,33),]
glm4v[which.min(glm4v[,2]),]
glm4=glm(y~prevworkplace+weight+height+retcat,binomial,data=traind1)
summary(glm4)

dev4 = 2*(logLik(glm4) - logLik(glm3))
pchisq(dev4, df=1, lower.tail=FALSE)

#model with 5 variables
names(traind1)[1:33]
glm5v = zeros(33,2)
colnames(glm5v) = c("Name", "BIC")

for(i in 1:33){
  if(i == 19){
    next
  }
  if(i == 26){
    next
  }
  if(i == 25){
    next
  }
  if(i == 16){
    next
  }
  x = colnames(traind1)[i]
  y1= as.formula(paste("y~prevworkplace+weight+height+retcat+", x))
  glmM=glm(y1,binomial,data=traind1)
  glm5v[i,1] = x
  glm5v[i,2] = BIC(glmM)
}

glm5v = glm5v[-c(19,26,25,16,33),]
glm5v[which.min(glm5v[,2]),]
glm5=glm(y~prevworkplace+weight+height+retcat+transport,binomial,data=traind1)
summary(glm5)

dev5 = 2*(logLik(glm5) - logLik(glm4))
pchisq(dev5, df=1, lower.tail=FALSE)


#model with 6 variables
names(traind1)[1:33]
glm6v = zeros(33,2)
colnames(glm6v) = c("Name", "BIC")

for(i in 1:33){
  if(i == 19){
    next
  }
  if(i == 26){
    next
  }
  if(i == 25){
    next
  }
  if(i == 16){
    next
  }
  if(i == 15){
    next
  }
  x = colnames(traind1)[i]
  y1= as.formula(paste("y~prevworkplace+weight+height+retcat+transport+", x))
  glmM=glm(y1,binomial,data=traind1)
  glm6v[i,1] = x
  glm6v[i,2] = BIC(glmM)
}

glm6v = glm6v[-c(19,26,25,16,33,15),]
glm6v[which.min(glm6v[,2]),]
glm6=glm(y~prevworkplace+weight+height+retcat+transport+appexp,binomial,data=traind1)
summary(glm6)

dev6 = 2*(logLik(glm6) - logLik(glm5))
pchisq(dev6, df=1, lower.tail=FALSE)

#model with 7 variables
names(traind1)[1:33]
glm7v = zeros(33,2)
colnames(glm7v) = c("Name", "BIC")

for(i in 1:33){
  if(i == 19){
    next
  }
  if(i == 26){
    next
  }
  if(i == 25){
    next
  }
  if(i == 16){
    next
  }
  if(i == 15){
    next
  }
  if(i == 23){
    next
  }  
  x = colnames(traind1)[i]
  y1= as.formula(paste("y~prevworkplace+weight+height+retcat+transport+appexp+", x))
  glmM=glm(y1,binomial,data=traind1)
  glm7v[i,1] = x
  glm7v[i,2] = BIC(glmM)
}

glm7v = glm7v[-c(19,26,25,16,33,15,23),]
glm7v[which.min(glm7v[,2]),]
glm7=glm(y~prevworkplace+weight+height+retcat+transport+appexp+extcourse,binomial,data=traind1)
summary(glm7)

dev7 = 2*(logLik(glm7) - logLik(glm6))
pchisq(dev7, df=1, lower.tail=FALSE)

#model with 8 variables
names(traind1)[1:33]
glm8v = zeros(33,2)
colnames(glm8v) = c("Name", "BIC")

for(i in 1:33){
  if(i == 19){
    next
  }
  if(i == 26){
    next
  }
  if(i == 25){
    next
  }
  if(i == 16){
    next
  }
  if(i == 15){
    next
  }
  if(i == 23){
    next
  }  
  if(i == 30){
    next
  }    
  x = colnames(traind1)[i]
  y1= as.formula(paste("y~prevworkplace+weight+height+retcat+transport+appexp+extcourse+", x))
  glmM=glm(y1,binomial,data=traind1)
  glm8v[i,1] = x
  glm8v[i,2] = BIC(glmM)
}

glm8v = glm8v[-c(19,26,25,16,33,15,23,30),]
glm8v[which.min(glm8v[,2]),]
glm8=glm(y~prevworkplace+weight+height+retcat+transport+appexp+extcourse+extra,binomial,data=traind1)
summary(glm8)

dev8 = 2*(logLik(glm8) - logLik(glm7))
pchisq(dev8, df=1, lower.tail=FALSE)

#model with 9 variables
names(traind1)[1:33]
glm9v = zeros(33,2)
colnames(glm9v) = c("Name", "BIC")

for(i in 1:33){
  if(i == 19){
    next
  }
  if(i == 26){
    next
  }
  if(i == 25){
    next
  }
  if(i == 16){
    next
  }
  if(i == 15){
    next
  }
  if(i == 23){
    next
  }  
  if(i == 30){
    next
  }  
  if(i == 6){
    next
  }    
  x = colnames(traind1)[i]
  y1= as.formula(paste("y~prevworkplace+weight+height+retcat+transport+appexp+extcourse+extra+", x))
  glmM=glm(y1,binomial,data=traind1)
  glm9v[i,1] = x
  glm9v[i,2] = BIC(glmM)
}

glm9v = glm9v[-c(19,26,25,16,33,15,23,30,6),]
glm9v[which.min(glm9v[,2]),]
glm9=glm(y~prevworkplace+weight+height+retcat+transport+appexp+extcourse+extra+med,binomial,data=traind1)
summary(glm9)

dev9 = 2*(logLik(glm9) - logLik(glm8))
pchisq(dev9, df=1, lower.tail=FALSE)


## final model - without salary
withoutSalary = glm(y~prevworkplace+weight+height+retcat+transport+appexp+extcourse+extra,binomial,data = traind1)
summary(withoutSalary)

#Save Model Female Without Salary
saveRDS(withSalary,"C:/Users/azeema/Desktop/Azeem/MSc/DS/PRoject/reaquadataset/LogisticFemaleWithoutSal.rds")


## evaluate model
predictWOS<-predict(withoutSalary, newdata=testd1, type="response")
accuracyWOS<-table(predictWOS>.5,testd1$y)
addmargins(accuracyWOS)

pred<-ifelse(predictWOS>.5,1,0)
table(pred,testd1$y)
## Accuracy
sum(diag(accuracyWOS))/sum(accuracyWOS)
