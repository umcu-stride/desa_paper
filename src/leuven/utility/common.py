def flatten2list(object) -> list:
    """ This function flattens objects in a nested structure and retu"""
    gather = []
    for item in object:
        if isinstance(item, (list, set)):
            gather.extend(flatten2list(item))
        else:
            gather.append(item)
    return gather

def flatten2set(object) -> set:
    """ This function flattens objects in a nested structure and returns a set"""

    return set(flatten2list(object))