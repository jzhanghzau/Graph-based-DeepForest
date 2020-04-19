# first line: 67
@men.cache

def get_data():
    data = load_svmlight_files("labeledBow.feat",24999)
    return data[0],data[1]
