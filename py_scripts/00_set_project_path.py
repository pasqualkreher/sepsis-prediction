import os
# This script write the path to the "mimic-iii-master" in to "./files/project_path.txt"
# its used especially for the func_* scripts to make them executable from the py_scripts and
# the notebooks folder
def set_project_path():
    parent_dir = os.getcwd()
    project_path = os.path.join(parent_dir, "files")
    txt_name = 'project_path.txt'
    if txt_name in os.listdir(project_path):
        os.remove(os.path.join(parent_dir, "files", txt_name))
    with open(os.path.join(parent_dir, "files","project_path.txt"), 'w') as f:
        f.write(parent_dir)
        print(parent_dir)
def main():
    set_project_path()

if __name__ == '__main__':
    main()


