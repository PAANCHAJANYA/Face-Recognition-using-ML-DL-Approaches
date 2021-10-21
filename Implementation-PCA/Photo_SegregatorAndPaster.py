import shutil
for i in []:
    for j in ["HIGH", "LOW", "MEDIUM"]:
        for k in ["WITH", "WITHOUT"]:
            for l in range (1, 101):
                if k == "WITH":
                                p = "1"
                if k == "WITHOUT":
                                p = "0"
                shutil.move("C:\\Users\\krish\\Desktop\\Krishna Work\\Face Detection and Recognition\\Database\\images\\S"+i+"\\"+j+"\\"+k+"\\"+"S"+i+"_0_"+j[0]+"_"+p+"_"+(str(l)).zfill(3)+"_0.jpg", "C:\\Users\\krish\\Desktop\\Krishna Work\\Face Detection and Recognition\\Train_Dataset")
