import pandas as pd
divider = '--------------------------'


def _convert_log_to_dicts(folder_name):
    data = []
    line = "??"
    with open(folder_name+'/debug.log', 'r') as f:
        while line != "":
            while divider not in line:
                line = f.readline()
            line = f.readline()
            epoch_data = dict()
            while divider not in line:
                attribute = line.split()
                epoch_data[attribute[0]] = attribute[1]
                line = f.readline()
            line = f.readline()
            avg = 0
            for k in (k for k in epoch_data.keys() if 'SuccessRate' in k):
                avg += float(epoch_data[k])
            epoch_data['AvgSuccessRate'] = avg / len([k for k in epoch_data.keys() if 'SuccessRate' in k])
            data.append(epoch_data)
    return data


def _extract_columns(data):
    columns = set()
    for d in data:
        columns |= set(d.keys())
    return columns


def convert(folder_name='./'):
    data = _convert_log_to_dicts(folder_name)
    columns = _extract_columns(data)
    df = pd.DataFrame(columns=columns)
    for d in data:
        df = df.append(d, ignore_index=True)
    df.to_csv(folder_name+'/converted_progress.csv', index=False)
