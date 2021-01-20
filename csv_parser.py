from csv import reader


def parse_data(data):
    with open(data) as csv_file:
        csv_reader = reader(csv_file, delimiter=',')
        line_count = 0
        count = 0
        unique = set()
        label_list = list()
        main_list = list()
        for line in csv_reader:
            if line_count != 0:
                value_list = list()
                [value_list.append(float(num))
                 for num in line if num is not line[-1]]
                if line[-1] not in unique:
                    unique.add(line[-1])
                    count +=1
                value_list.append(len(unique))
                main_list.append(value_list)
                label_list.append(count)

            line_count += 1

        return main_list, label_list
