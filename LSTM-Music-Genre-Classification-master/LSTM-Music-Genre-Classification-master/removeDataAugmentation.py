import os 

archivos = os.listdir("./gtzan/_train/") 


for file_name in archivos:
    
   if  "noise" in file_name:
       os.remove("./gtzan/_train/"+ file_name)
   

