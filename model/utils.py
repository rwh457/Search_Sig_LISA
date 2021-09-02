import os
import fnmatch
import csv


def print_dict(d: dict,  ncol: int, prex: str = ''):
    temp = [f"{k}: {v}" for k, v in d.items()]
    print(prex+('\n'+prex).join([', '.join(temp[i:: ncol]) for i in range(ncol)]))


def writer_row(p, filename, mode, contentrow):
    with open(p / filename, mode) as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(contentrow)


def ffname(path, pat):
    """Return a list containing the names of the files in the directory matches PATTERN.

    'path' can be specified as either str, bytes, or a path-like object.  If path is bytes,
        the filenames returned will also be bytes; in all other circumstances
        the filenames returned will be str.
    If path is None, uses the path='.'.
    'Patterns' are Unix shell style:
        *       matches everything
        ?       matches any single character
        [seq]   matches any character in seq
        [!seq]  matches any char not in seq
    Ref: https://stackoverflow.com/questions/33743816/how-to-find-a-
        filename-that-contains-a-given-string
    """
    return [filename
            for filename in os.listdir(path)
            if fnmatch.fnmatch(filename, pat)]
