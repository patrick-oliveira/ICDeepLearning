import shutil, os

def main():
    path = 'C:\\Users\\Patrick\\Documents\\GitHub\\ICDeepLearning\\Data\\Backup\\Originais\\512'
    subjects = os.listdir(path)

    for subject in subjects:
        files = [x for x in os.listdir(path+'\\'+subject) if '.txt' in x]
        for state in [1, 2, 3, 4, 5]:
            subfiles = [x for x in files if 'state'+str(state) in x]
            for f in subfiles:
                shutil.move(path+'\\'+subject+'\\'+f, path+'\\'+subject+'\\'+str(state))

    for subject in subjects:
        for state in [1, 2, 3, 4, 5]:
            new_path = path+'\\'+subject+'\\'+str(state)
            files = os.listdir(new_path)
            for i in range(len(files)):
                os.rename(new_path+'\\'+files[i], new_path+'\\'+str(i)+'.txt')

if __name__ == '__main__':
    main()
    