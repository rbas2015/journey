library(readxl)

PATH_NETMIS <- "C:/More Documents/Data/new_tfl/NetMis/OctNovDec13_AcademicDataset/"

FILE_NAME <- "27-Oct-2013"
xs <- excel_sheets(paste(PATH_NETMIS, paste(FILE_NAME, ".xlsx", sep = ""), sep = ""))
num_sheets <- length(xs)
for (i in 4:num_sheets) {
  cat(paste("Reading sheet", xs[i], "\n"))
  f <- read_excel(paste(PATH_NETMIS, FILE_NAME, ".xlsx", sep = ""), sheet = i)
  write.csv(f, file = paste(PATH_NETMIS, xs[i], ".csv", sep = ""), row.names = FALSE)
}
cat("Done!\n")
