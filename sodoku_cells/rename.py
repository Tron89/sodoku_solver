import os

i = 0
none = []
for actual_path, directories, files in os.walk(os.getcwd()):
    if directories == none:
        y = 0
        for file in files:
            file_path = os.path.join(actual_path, file)
            os.rename(file_path, os.path.join(actual_path, f"{str(i)}_{str(y)}.jpg"))
            y += 1
        i += 1
    
input("Mision completed")