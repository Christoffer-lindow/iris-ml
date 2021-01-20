def seperate_classes(input_list):
    seperated_classes = dict()
    for line in input_list:
        if line[-1] not in seperated_classes:
            seperated_classes[line[-1]] = list()
        seperated_classes[line[-1]].append(line[:-1])
    return seperated_classes

def get_values_for_col(col,input_list):
    return_list = list()
    for list_vals in input_list:
            return_list.append(list_vals[col])
    return return_list