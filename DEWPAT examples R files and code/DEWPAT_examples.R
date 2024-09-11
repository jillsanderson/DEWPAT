### CODE TO COMPLETE DEWPAT EXAMPLES 

setwd("")




########################################## Example 1
beetle_dat <- read.csv("beetle_complexity.csv", row.names = 1, header = TRUE,na.strings = c("", "NA"))

pcaOutput <- prcomp(beetle_dat, center = TRUE, scale = TRUE)
summary(pcaOutput)
unclass(pcaOutput)
biplot(pcaOutput, xlab="PC1 (54%)", ylab = "PC2 (33%)", main = "biplot", cex = 0.5)







########################################## Example 2

# Colour Contrast (Euclidean Distance)
onecolourdata <- read.csv("anole_dew_lab_2cluster.csv",row.names=1,header = TRUE,na.strings = c("", "NA"))

names(onecolourdata)

#make new objects with the variables of interest
L1 <- onecolourdata[,1] 
names(L1) <- row.names(onecolourdata) 
a1 <- onecolourdata[,2] 
names(a1) <- row.names(onecolourdata) 
b1 <- onecolourdata[,3] 
names(b1) <- row.names(onecolourdata) 
L2 <- onecolourdata[,4] 
names(L2) <- row.names(onecolourdata) 
a2 <- onecolourdata[,5] 
names(a2) <- row.names(onecolourdata) 
b2 <- onecolourdata[,6] 
names(b2) <- row.names(onecolourdata) 

# Function to calculate Delta E (colour contrast)
delta_e_ab <- function(L1, a1, b1, L2, a2, b2) {
  sqrt((L1 - L2)^2 + (a1 - a2)^2 + (b1 - b2)^2)
}

# Calculate Delta E_ab
delta_e <- delta_e_ab(L1, a1, b1, L2, a2, b2)

#Add a column 
onecolourfinish <- cbind(onecolourdata, delta_e)

#Export
write.csv(onecolourfinish, file ="anole_lab_coldist.csv")


# create pie charts
# RGB to Hex
anole_rgb <- read.csv("anole_dew_col_2cluster.csv")
names(anole_rgb)
rgb2hex <- function(R, G, B) {rgb(R, G, B, maxColorValue = 255)}
anole_rgb$Hex <- rgb2hex(anole_rgb[3:5])

write.csv(anole_rgb, "rgb2hex_dew.csv")

# Pie Charts #
anole_rgb2 <- read.csv("rgb2hex_dew.csv")
names(anole_rgb2)

#break up dataset into data for individual species
splist = unique(anole_rgb2$species) # list of unique species names
anole.pi = list()
for(i in 1:length(splist)) anole.pi[[i]] = anole_rgb2[anole_rgb2$species == splist[i],]
names(anole.pi) = splist
# anole.polygons is now a list in which each element contains the polygon (or, in many cases, multiple polygons) for a single species
# have a look at the structure
str(anole.pi, max.level=1)
# check out the first few elements
anole.pi[[1]]
anole.pi[[1]]$Hex

plot<- ggplot(anole.pi[[1]], aes(x="", y=percent, fill=as.factor(cluster))) +
  geom_bar(width = 1, stat = "identity", fill =anole.pi[[1]]$Hex) +
  coord_polar("y", start=0) +
  labs(x = "", y = "") 
plot + theme_bw() + theme(axis.text.x=element_blank(), axis.ticks.x=element_blank(), axis.text.y=element_blank(), axis.ticks.y=element_blank())

#loop the plots and put them in object pi.plots
pi.plots = list()
for(i in 1:length(anole.pi)) pi.plots[[i]] = ggplot(anole.pi[[i]], aes(x="", y=percent, fill=as.factor(cluster))) +
  geom_bar(width = 1, stat = "identity", fill =anole.pi[[i]]$Hex) +
  coord_polar("y", start=0) +
  labs(x = "", y = "") +
  theme_bw() + theme(axis.text.x=element_blank(), axis.ticks.x=element_blank(), axis.text.y=element_blank(), axis.ticks.y=element_blank())

pi.plots[1]







########################################## Example 3
library(ggplot2)

# Bee only
flower_dat_bee <- read.csv("flowers_bee.csv", header = TRUE,na.strings = c("", "NA"))
names(flower_dat_bee)

flower_dat_bee$GPC_stnd <- 
  (flower_dat_bee$Global.Patch.Covariance - min(flower_dat_bee$Global.Patch.Covariance)) /
  (max(flower_dat_bee$Global.Patch.Covariance) - min(flower_dat_bee$Global.Patch.Covariance))


#plot
theme <-  theme_bw() + theme(legend.direction="horizontal", legend.position=c(0.9,1.034), legend.title=element_blank(), text = element_text(size = 19))
flower <- ggplot(flower_dat_bee, aes(x=species, y=GPC_stnd)) + 
  geom_violin(trim=FALSE, fill="gray")+
  labs(title="", x = "", y = "Global Heterogeneity")
flower + theme


lm <- lm(GPC_stnd ~ species, data=flower_dat_bee)
summary(lm) # p = 0.0316, adjusted R squared = 0.1566

anova_result <- aov(GPC_stnd ~ species, data = flower_dat_bee)
summary(anova_result) # F-statistic = 186.2, p < 0.001

# Vis only
flower_dat_vis <- read.csv("flowers_vis.csv", header = TRUE,na.strings = c("", "NA"))
flower_dat_vis$GPC_stnd <- 
  (flower_dat_vis$Global.Patch.Covariance - min(flower_dat_vis$Global.Patch.Covariance)) /
  (max(flower_dat_vis$Global.Patch.Covariance) - min(flower_dat_vis$Global.Patch.Covariance))


#plot
theme <-  theme_bw() + theme(legend.direction="horizontal", legend.position=c(0.9,1.034), legend.title=element_blank(), text = element_text(size = 19))
flower <- ggplot(flower_dat_vis, aes(x=species, y=GPC_stnd)) + 
  geom_violin(trim=FALSE, fill="gray")+
  labs(title="", x = "", y = "Global Heterogeneity")
flower + theme

lm <- lm(GPC_stnd ~ species, data=flower_dat_vis)
summary(lm) # p = 3.24e-12, adjusted R squared = 0.8895

anova_result <- aov(GPC_stnd ~ species, data = flower_dat_vis)
summary(anova_result) # F-statistic = 186.2, p < 0.001






