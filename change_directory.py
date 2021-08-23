with open('output/test_paths.txt', 'r') as file:
    # read a list of lines into data
    directories = file.readlines()
    new_directories = []

# fill with modified directories
for directory in directories:
    #new_directory = directory.replace('/content/drive/MyDrive/MSc Applied Artificial Intelligence (Cranfield University)/IRP/Neural Network','/home/daniel/Escritorio/MSc AAI/IRP/Neural Network')
    new_directory = directory.replace('/content/drive/MyDrive/IRP/Neural Network','/home/daniel/Escritorio/MSc AAI/IRP/Neural Network')
    new_directories.append(new_directory)

# and write everything back
with open('output/test_paths.txt', 'w') as file:
    file.writelines(new_directories)