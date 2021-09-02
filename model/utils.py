import csv


def print_dict(d: dict,  ncol: int, prex: str = ''):
    temp = [f"{k}: {v}" for k, v in d.items()]
    print(prex+('\n'+prex).join([', '.join(temp[i:: ncol]) for i in range(ncol)]))


def writer_row(p, filename, mode, contentrow):
    with open(p / filename, mode) as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(contentrow)
