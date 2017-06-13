using ExcelReaders

PATH_NETMIS = "C:\\More Documents\\Data\\new_tfl\NetMis\\JanFeb14_AcademicDataset\\";
FILE_NAME = "09-Feb-2014.xlsx";
data = openxl(PATH_NETMIS * FILE_NAME);