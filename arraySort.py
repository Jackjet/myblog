def ascending_order1():
    intStr = raw_input('Enter a array of number:')
    s1 = intStr.split(" ")
    arr=[]
    for a in s1:
       arr.append(int(a))
    if len(arr)== 0:
       return -1    
    
    j=0
    while j in range (0,len(arr)):
      Max=arr[j]
      for i in range (j,len(arr)):
        if arr[i] > Max:
            Max = arr[i]
            arr[i]= arr[j]
            arr[j]= Max
        i +=1
      j+=1
    print arr
    return 0



a = [3,6,9,1,88,12]
ascending_order1()
