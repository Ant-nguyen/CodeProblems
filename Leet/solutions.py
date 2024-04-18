
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


# Leet Code 11: Container With Most Water
def func (arr):
    l = 0
    r = len(arr)-1
    w = r - l
    low_height = min(arr[l],arr[r])
    max_area = low_height * w
    while w != 0:
        # left?
        if low_height == arr[l]:
            l += 1
            w -= 1
            low_height = min(arr[l],arr[r])
            area = low_height*w
            if area > max_area:
                max_area = area
        # right
        else:
            r -= 1
            w -= 1
            low_height = min(arr[l],arr[r])
            area = low_height*w
            if area > max_area:
                max_area = area
    return max_area


# LeetCode 1679. Max Number of k-sum Pairs

# Brute force way using HashTable:

# class Solution:
#     def maxOperations(self, nums: List[int], k: int) -> int:
#         hash = {}
#         count = 0
#         for ind,num in enumerate(nums):
#             if num in hash:
#                 hash[num].pop(0)
#                 if len(hash[num]) == 0:
#                     del hash[num]
#                 count += 1
#             else:
#                 if k-num in hash:
#                     hash[k-num].append(ind)
#                 else:
#                     hash[k-num] = [ind]
#         return count

# Solution using two Pointers:
def maxOperations(nums, k: int) -> int:
    nums.sort()
    left,right,count = 0,len(nums)-1,0
    while left < right:
        if nums[left] + nums[right] == k:
            count+= 1
            left += 1
            right -= 1
            continue
        elif nums[left] + nums[right] > k:
            right -= 1
        else:
            left += 1
    return count



# LeetCode: 1456. Maximum Number of Vowels in a Substring of Given Length
# Sliding window
def maxVowels(self, s: str, k: int) -> int:
    vowels = {"a","e","i","o","u"}
    count = sum([1 for n in s[0:k] if n in vowels])
    max = count
    # starting at k essentially makes i the end of the window and i-k the front!
    for i in range(k,len(s)):
        if count == k:
            return k
        if s[i] in vowels:
            count += 1
        if s[i-k] in vowels:
            count -= 1
        if count > max:
            max = count
    return max