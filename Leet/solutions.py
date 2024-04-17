
#Leetcode 334

# a way to remember previous best "pair" AND continue forward looking for possible smallest i,j pairing, old 
# j essentially is the representation of the last potential i,j sequence. If something gets pass the i check and 
# is smaller than j, they are the new lower i,j sequence and anything that would have met the condition of 
# the previous i,j,k potential would also be met with the new currest best i,j sequence and any possible k

def increasingTriplet(nums) -> bool:
    i = float("inf")
    j = float("inf")
    for n in nums:
        if n <= i:
            i = n
        elif n <= j:
            j = n
        else:
            return True
    return False

test = [20,100,10,12,5,13]
print(increasingTriplet(test))


# LeetCode 26

# changes need to happen in place
# esentially just look for each unique value to add to the next earliest position on the list
def removeDuplicates(nums) -> int:
    i = 0
    for j in range(1,len(nums)):
        if nums[i] != nums[j]:
            i+= 1
            nums[i]=nums[j]
    return i+1


# Leet code 128.  Longest Consecutive Sequence
def longestConsecutive( nums) -> int:
    # edge case catch
    if len(nums)==0:
        return 0
    sorts = sorted(nums)
    hash = {num: 1 for num in nums}
    for i,num in enumerate(sorts):
        #Check to see if previous number is sequential
        if sorts[i-1] == num-1:
            # take the previous numbers sequential total and add to current
            hash[num] += hash[num-1]
    return max(hash.values())



