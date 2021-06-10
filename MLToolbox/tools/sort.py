def add_to_topk(lst, new_num, reverse=False):
    '''
    return position of where new point 'n' should be.
    '''
    # TODO(zhangfelixx@hotmail.com):add a new number to topk list, return None if new number
    # not amongs to top k; return position if new number amongs top k. Don't change original list
    for idx, element in enumerate(lst):
        if ((element > new_num) ^ reverse) and element != new_num:
            return idx
    return None

def add_to_topk_binary(lst, new_num, reverse=False):
    '''
    return position of where new point 'n' should be. Using binary search.
    '''
    left = 0
    right = len(lst) - 1
    pivot = (left + right) // 2

    def find_position():
        nonlocal left
        nonlocal right
        nonlocal pivot
        if lst[pivot] is None or lst[pivot] > new_num:
            right = pivot
            pivot = (left + right) // 2
        else:
            left = pivot
            pivot = (left + right) // 2

    def find_position_reverse():
        nonlocal left
        nonlocal right
        nonlocal pivot
        if lst[pivot] is None or lst[pivot] < new_num:
            right = pivot
            pivot = (left + right) // 2
        else:
            left = pivot
            pivot = (left + right) // 2


            # 1,2,3,4,5,5,5,6,7,8
            # 8,7,6,5,5,5,4,3,2,1
    while right - left > 1:
        if reverse is False:
            find_position()
        else:
            find_position_reverse()

    if reverse is False:
        if lst[right] is not None and lst[right] <= new_num:
            return None
        if lst[left] is not None and lst[left] <= new_num:
            return right
        return left

    if lst[right] is not None and lst[right] >= new_num:
        return None
    if lst[left] is not None and lst[left] >=new_num:
        return right
    return left
